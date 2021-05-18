import os
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import math
import numpy as np
import itertools
import sys

from src.model import LaserNet, build_lasernet_functional
from src.loss import LaserNetLoss, ClassLoss
from tensorflow import keras
from datetime import datetime

from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils, box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

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

def fill_boxes(input, indices, labels):
    gathered = tf.gather(labels, indices, axis=1, batch_dims=1)
    dense_boxes = corners_from_centers(gathered)
    classes = tf.where(tf.greater_equal(indices, 1), 1, 0)
    classes = classes[..., 992:1656]
    return input, {'classes': tf.concat([classes, tf.cast(input['input_laser'][..., 4], tf.int32)], -1)}#, "boxes": tf.concat([dense_boxes, tf.expand_dims(tf.cast(indices, tf.float32), -1)], -1)}

def transform(r, e, i, labels):
    polar_image = range_image_utils.compute_range_image_polar(tf.expand_dims(r[..., 0], 0), tf.expand_dims(e, 0), tf.expand_dims(i, 0))
    cloud = range_image_utils.compute_range_image_cartesian(polar_image, tf.expand_dims(e, 0))
    # print(cloud.get_shape().num_elements())
    flattened_cloud = tf.reshape(cloud[..., 0:2], [cloud.get_shape().num_elements() // 3, 2])
    bool_match = box_utils.is_within_box_2d(flattened_cloud, labels)
    # Pad to add a zero-index indicating no box match
    bool_match = tf.pad(bool_match, [[0, 0], [1, 0]], constant_values=False)
    print(r.get_shape()[:-1])
    indices = tf.reshape(tf.argmax(bool_match, axis=-1), r.get_shape()[:-1])

    azimuth = polar_image[0, ..., 0]
    height = tf.math.sin(polar_image[0, ...,1]) * tf.math.maximum(polar_image[0, ..., 2], 0)
    range = polar_image[0, ..., 2]
    intensity = r[..., 1]
    mask = tf.greater_equal(r[..., 0], 0)
    mask = tf.where(mask, 1.0, 0.0)
    return {'input_laser': tf.stack([azimuth, height, range, intensity, mask], -1)[..., 992:1656, :], 'input_xy': tf.squeeze(cloud[..., 0:2], axis=0)[..., 992:1656, :]}, indices, tf.concat([[[0, 0, 0, 0, 0]], labels], 0)

tensor_ds = tf.data.experimental.load('/home/alex/alex-usb/compressed_ds', (
                                                    tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                                                    tf.TensorSpec(shape=[16], dtype=tf.float32), 
                                                    tf.TensorSpec(shape=[64], dtype=tf.float32), 
                                                    tf.TensorSpec(shape=None, dtype=tf.float32)),
                                                    compression="GZIP"
                                                    ).shuffle(1024).prefetch(128)
tensor_ds = tensor_ds.map(shape, num_parallel_calls=32)
tensor_ds = tensor_ds.map(transform, num_parallel_calls=32)
tensor_ds = tensor_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=128, drop_remainder=True))
final_ds = tensor_ds.map(fill_boxes, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
logs = "logs/" + run_id
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, update_freq=10)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('checkpoints', run_id, 'checkpoint-{epoch:02d}.ckpt'),
    monitor='loss',
    mode='min',
    save_best_only=False)
# original
means = [-2.5876644, -2.120523 , 18.519848 , 15.371667 ,  0.834153 ]
variances = [2.0690569e-01, 1.2527465e+01, 2.7277847e+02, 4.0037647e+05, 0.27668]

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, update_freq=5)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/alex/checkpoints/checkpoint-{epoch:02d}.ckpt',
    monitor='loss',
    mode='min',
    save_best_only=False)
# original
means = [-2.5962167 , -1.8928711 , 16.83029   ,  0.524463  ,  0.91721344]
variances = [3.2898679e+00, 6.5204339e+00, 1.3595302e+02, 2.5995129e+03, 1.5871947e-01]
# means = [-2.5825715 , -1.4563112 , 12.411071  , -0.03784659,  0.78327984]
# variances = [ 0.2056163 ,  5.4643445 , 81.9744    ,  0.13470726,  0.3864727 ]
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = tf.keras.models.load_model('/home/alex/lasernet-waymo/checkpoints/20210511-011201/checkpoint-14.ckpt', custom_objects={'ClassLoss': ClassLoss})

file_writer = tf.summary.create_file_writer(logs + '/images')
def evaluate_images(epoch, logs):
  data, label = next(iter(final_ds.take(1)))
  test_pred_raw = model.predict(data)
  test_pred = tf.keras.layers.Softmax()(test_pred_raw)[..., 1:]
  test_label = tf.int32.max * tf.expand_dims(label['classes'], -1)
  print(data['input_laser'][0, ..., 0:1])
  with file_writer.as_default():
    tf.summary.image("Prediction", test_pred, step=epoch, max_outputs=4)
    tf.summary.image("Label", test_label, step=epoch, max_outputs=4)
    tf.summary.image("Azimuth", data['input_laser'][..., 0:1], step=epoch, max_outputs=4)
    tf.summary.image("Height", data['input_laser'][..., 1:2], step=epoch, max_outputs=4)
    tf.summary.image("Range", data['input_laser'][..., 2:3], step=epoch, max_outputs=4)
    tf.summary.image("Intensity", data['input_laser'][..., 3:4], step=epoch, max_outputs=4)
    tf.summary.image("Mask", data['input_laser'][..., 4:5], step=epoch, max_outputs=4)

model.fit(final_ds, epochs=10, steps_per_epoch=10, callbacks = [tboard_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_images)])