import os
import tensorflow as tf
import numpy as np

from src.model import build_lasernet_functional
from src.loss import ClassLoss, BoxLoss
from tensorflow import keras
from datetime import datetime

from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils, box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

def parse_frame_parallel(data):
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())
    top_image = frame_utils.parse_range_image_and_camera_projection(frame)[0][open_dataset.LaserName.TOP][0]
    range_image_tensor = tf.convert_to_tensor(top_image.data)
    frame.context.laser_calibrations.sort(key=lambda laser: laser.name)
    c = frame.context.laser_calibrations[open_dataset.LaserName.TOP-1]
    extrinsic = tf.convert_to_tensor(c.extrinsic.transform)
    beam_inclinations = tf.convert_to_tensor(c.beam_inclinations)
    vehicle_labels = []
    for label in frame.laser_labels:
        if label.type == label.TYPE_VEHICLE:
            vehicle_labels.append([label.box.center_x, label.box.center_y, label.box.length, label.box.width, label.box.heading])
    num_labels = len(vehicle_labels)
    vehicle_labels = tf.convert_to_tensor(vehicle_labels)
    return range_image_tensor, extrinsic, beam_inclinations, vehicle_labels, [num_labels, 5]

def shape(r, e, i, labels):
    return tf.reshape(r, [64, 2650, 4])[:, 0:2648, :], tf.reshape(e, [4,4]), tf.reshape(i, [64]), tf.reshape(labels, [-1, 5])

def corners_from_centers(center_boxes):
    centers = center_boxes[..., 0:2]
    length = center_boxes[..., 2]
    width = center_boxes[..., 3]
    orientation = center_boxes[..., 4]
    orientation_mat_flat = tf.stack([tf.math.cos(orientation), -1 * tf.math.sin(orientation), 
                                    tf.math.sin(orientation), tf.math.cos(orientation)], axis=-1)
    orientation_mat = tf.reshape(orientation_mat_flat, tf.concat([orientation_mat_flat.get_shape()[:-1], [2, 2]], 0))
    corner1 = centers + 0.5 * tf.squeeze(orientation_mat @ tf.expand_dims(tf.stack([length, width], axis=-1), -1), -1)
    corner2 = centers + 0.5 * tf.squeeze(orientation_mat @ tf.expand_dims(tf.stack([length, -1 * width], axis=-1), -1), -1)
    corner3 = centers + 0.5 * tf.squeeze(orientation_mat @ tf.expand_dims(tf.stack([-1 * length, -1 * width], axis=-1), -1), -1)
    corner4 = centers + 0.5 * tf.squeeze(orientation_mat @ tf.expand_dims(tf.stack([-1 * length, width], axis=-1), -1), -1)
    # B H W 8
    return tf.concat([corner1, corner2, corner3, corner4], -1)

def fill_boxes(input, nlz, indices, labels):
    gathered = tf.gather(labels, indices, axis=1, batch_dims=1)
    dense_boxes = corners_from_centers(gathered)
    classes = tf.where(tf.greater_equal(indices, 1), 1, 0)
    classes = classes[..., 0:2648]
    class_labels = tf.stack([classes, tf.cast(input['input_laser'][..., 4], tf.int32), tf.cast(nlz, tf.int32)], -1)
    box_labels = tf.concat([tf.expand_dims(tf.cast(classes, tf.float32), -1), dense_boxes, tf.expand_dims(tf.cast(indices, tf.float32), -1)], -1)
    return input, {'classes': class_labels, 'boxes': box_labels}

def transform(r, e, i, labels):
    polar_image = range_image_utils.compute_range_image_polar(tf.expand_dims(r[..., 0], 0), tf.expand_dims(e, 0), tf.reverse(tf.expand_dims(i, 0), [-1]))
    cloud = range_image_utils.compute_range_image_cartesian(polar_image, tf.expand_dims(e, 0))
    flattened_cloud = tf.reshape(cloud[..., 0:2], [cloud.get_shape().num_elements() // 3, 2])
    bool_match = box_utils.is_within_box_2d(flattened_cloud, labels)
    # Pad to add a zero-index indicating no box match
    bool_match = tf.pad(bool_match, [[0, 0], [1, 0]], constant_values=False)
    print(r.get_shape()[:-1])
    indices = tf.reshape(tf.argmax(bool_match, axis=-1), r.get_shape()[:-1])

    azimuth = polar_image[0, ..., 0]
    correction = tf.atan2(e[..., 1, 0], e[..., 0, 0])
    azimuth = azimuth + correction
    height = tf.math.sin(polar_image[0, ...,1]) * tf.math.maximum(polar_image[0, ..., 2], 0)
    range = polar_image[0, ..., 2]
    intensity = r[..., 1]
    nlz = r[..., 3]
    mask = tf.greater_equal(r[..., 0], 0)
    mask = tf.where(mask, 1.0, 0.0)
    input_laser = tf.stack([azimuth, height, range, intensity, mask], -1)
    input_xyz = tf.squeeze(cloud[..., 0:], axis=0)
    return {'input_laser': input_laser[..., 0:2648, :], 'input_xyz': input_xyz[..., 0:2648, :]}, nlz[..., 0:2648], indices, tf.concat([[[0, 0, 0, 0, 0]], labels], 0)

def custom_reader_func(datasets: tf.data.Dataset):
    datasets = datasets.shuffle(20)
    return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=20)

datasets = []
for i in range(0, 20):
    ds = tf.data.experimental.load('/home/alex/dataset-drive/ds_sharded/shard_%d'%i, (
                            tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                            tf.TensorSpec(shape=[16], dtype=tf.float32), 
                            tf.TensorSpec(shape=[64], dtype=tf.float32), 
                            tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')
    datasets.append(ds)

tensor_ds = tf.data.experimental.sample_from_datasets(datasets).shuffle(1000)
tensor_ds = tensor_ds.map(shape, num_parallel_calls=32)
tensor_ds = tensor_ds.map(transform, num_parallel_calls=32)
tensor_ds = tensor_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=16, drop_remainder=True))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
final_ds = tensor_ds.map(fill_boxes, num_parallel_calls=tf.data.AUTOTUNE).with_options(options).prefetch(2)

validation_ds = tf.data.experimental.load('/home/alex/dataset-drive/validation', (
                                            tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                                            tf.TensorSpec(shape=[16], dtype=tf.float32), 
                                            tf.TensorSpec(shape=[64], dtype=tf.float32), 
                                            tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')

validation_ds = validation_ds.map(shape, num_parallel_calls=32)
validation_ds = validation_ds.map(transform, num_parallel_calls=32)
validation_ds = validation_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=16, drop_remainder=True))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
validation_ds = validation_ds.map(fill_boxes, num_parallel_calls=tf.data.AUTOTUNE).with_options(options).prefetch(2)

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
logs = "logs/" + run_id
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, update_freq=10)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('checkpoints', run_id, 'checkpoint-{epoch:02d}.ckpt'),
    monitor='loss',
    mode='min',
    save_best_only=False)

means = [8.6231437e-04, -2.0784380e+00,  1.7843075e+01,  9.4569902e+00, 7.8301370e-01]
variances = [2.0686106e-01, 1.2035955e+01, 2.8855536e+02, 2.5339364e+05, 1.6990326e-01]
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = build_lasernet_functional(means=means, variances=variances)
schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 1200, 0.99)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=1)
model.compile(optimizer=optimizer, loss={'classes': ClassLoss(), 'boxes': BoxLoss()})

# print(logs + '/images')
file_writer = tf.summary.create_file_writer(logs + '/images')
repeated_ds = final_ds.take(1).cache()
def evaluate_images(epoch, logs):
    for data, label in repeated_ds:
        test_pred_raw = model.predict(data)[0]
        test_pred = tf.keras.layers.Softmax()(test_pred_raw)[..., 1:]
        test_label = tf.int32.max * label['classes'][..., 0:1]
        test_nlz = tf.cast(label['classes'][..., 2:] + 1, tf.float32) / 2.0
        with file_writer.as_default():
            tf.summary.image("Prediction", test_pred, step=epoch, max_outputs=4)
            tf.summary.image("Label", test_label, step=epoch, max_outputs=4)
            tf.summary.image("No Label Zone", test_nlz, step=epoch, max_outputs=4)
        break
            # Prints false if images are not written because epoch length not divisible by update_freq

model.fit(final_ds.take(9880), 
            epochs=40, 
            callbacks = [tboard_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_images), model_checkpoint_callback],
            validation_data=validation_ds, 
            validation_freq=4)