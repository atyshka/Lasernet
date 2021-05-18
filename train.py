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

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

import multiprocessing
from typing import Callable, Union, List
import signal


class PyMapper:
    """
    A class which allows for mapping a py_function to a TensorFlow dataset in parallel on CPU.
    """
    def __init__(self, map_function: Callable, number_of_parallel_calls: int):
        self.map_function = map_function
        self.number_of_parallel_calls = number_of_parallel_calls
        self.pool = multiprocessing.Pool(self.number_of_parallel_calls, self.pool_worker_initializer)

    @staticmethod
    def pool_worker_initializer():
        """
        Used to initialize each worker process.
        """
        # Corrects bug where worker instances catch and throw away keyboard interrupts.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def send_to_map_pool(self, element_tensor):
        """
        Sends the tensor element to the pool for processing.

        :param element_tensor: The element to be processed by the pool.
        :return: The output of the map function on the element.
        """
        result = self.pool.apply_async(self.map_function, (element_tensor,))
        mapped_element = result.get()
        return mapped_element

    def map_to_dataset(self, dataset: tf.data.Dataset,
                       output_types: Union[List[tf.dtypes.DType], tf.dtypes.DType] = tf.float32):
        """
        Maps the map function to the passed dataset.

        :param dataset: The dataset to apply the map function to.
        :param output_types: The TensorFlow output types of the function to convert to.
        :return: The mapped dataset.
        """
        def map_py_function(*args):
            """A py_function wrapper for the map function."""
            return tf.py_function(self.send_to_map_pool, args, output_types)
        return dataset.map(map_py_function, self.number_of_parallel_calls)


def map_py_function_to_dataset(dataset: tf.data.Dataset, map_function: Callable, number_of_parallel_calls: int,
                               output_types: Union[List[tf.dtypes.DType], tf.dtypes.DType] = tf.float32
                               ) -> tf.data.Dataset:
    """
    A one line wrapper to allow mapping a parallel py function to a dataset.

    :param dataset: The dataset whose elements the mapping function will be applied to.
    :param map_function: The function to map to the dataset.
    :param number_of_parallel_calls: The number of parallel calls of the mapping function.
    :param output_types: The TensorFlow output types of the function to convert to.
    :return: The mapped dataset.
    """
    py_mapper = PyMapper(map_function=map_function, number_of_parallel_calls=number_of_parallel_calls)
    mapped_dataset = py_mapper.map_to_dataset(dataset=dataset, output_types=output_types)
    return mapped_dataset

# file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/training/*.tfrecord')
# file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord')
# record_ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

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
    # tf.print(r.shape)
    # tf.print(e.shape)
    # tf.print(i.shape)
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
    return input, {'classes': tf.stack([classes, tf.cast(input['input_laser'][..., 4], tf.int32)], -1)}#, "boxes": tf.concat([dense_boxes, tf.expand_dims(tf.cast(indices, tf.float32), -1)], -1)}

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
    azimuth = tf.math.atan2(tf.math.sin(azimuth), tf.math.cos(azimuth))
    height = tf.math.sin(polar_image[0, ...,1]) * tf.math.maximum(polar_image[0, ..., 2], 0)
    range = polar_image[0, ..., 2]
    intensity = r[..., 1]
    mask = tf.greater_equal(r[..., 0], 0)
    mask = tf.where(mask, 1.0, 0.0)
    input_laser = tf.stack([azimuth, height, range, intensity, mask], -1)
    input_xy = tf.squeeze(cloud[..., 0:2], axis=0)
    correction = tf.atan2(e[..., 1, 0], e[..., 0, 0])
    input_laser = tf.roll(input_laser, tf.cast((correction / math.pi) * 2650 / 2, tf.int32), 1)
    input_xy = tf.roll(input_xy, tf.cast((correction / math.pi) * 2650 / 2, tf.int32), 1)
    indices = tf.roll(indices, tf.cast((correction / math.pi) * 2650 / 2, tf.int32), 1)
    return {'input_laser': input_laser[..., 992:1656, :], 'input_xy': input_xy[..., 992:1656, :]}, indices, tf.concat([[[0, 0, 0, 0, 0]], labels], 0)

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

tensor_ds = tf.data.experimental.sample_from_datasets(datasets).repeat().shuffle(10000)
# tensor_ds = tf.data.experimental.load('/home/alex/alex-usb/interleaved_ds', (
#                                                     tf.TensorSpec(shape=[678400], dtype=tf.float32), 
#                                                     tf.TensorSpec(shape=[16], dtype=tf.float32), 
#                                                     tf.TensorSpec(shape=[64], dtype=tf.float32), 
#                                                     tf.TensorSpec(shape=None, dtype=tf.float32)),
#                                                     compression="GZIP", reader_func=custom_reader_func
#                                                     ).repeat().shuffle(5000).prefetch(128)
# tensor_ds = tf.data.experimental.load('/home/alex/dataset-drive/ds_sharded/shard_0', (
#                         tf.TensorSpec(shape=[678400], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[16], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[64], dtype=tf.float32), 
#                         tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')
# tensor_ds = map_py_function_to_dataset(record_ds, parse_frame_parallel, 32, (tf.float32, tf.float32, tf.float32, tf.float32)).cache().shuffle(200).repeat()
tensor_ds = tensor_ds.map(shape, num_parallel_calls=32)
tensor_ds = tensor_ds.map(transform, num_parallel_calls=32)
tensor_ds = tensor_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=64, drop_remainder=True))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
final_ds = tensor_ds.map(fill_boxes, num_parallel_calls=tf.data.AUTOTUNE).with_options(options).prefetch(2)

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
logs = "logs/" + run_id
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, update_freq=10)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('checkpoints', run_id, 'checkpoint-{epoch:02d}.ckpt'),
    monitor='loss',
    mode='min',
    save_best_only=False)
# original
# means = [-2.5876644, -2.120523 , 18.519848 , 15.371667 ,  0.834153 ]
# variances = [2.0690569e-01, 1.2527465e+01, 2.7277847e+02, 4.0037647e+05, 0.27668]
means = [8.6231437e-04, -2.0784380e+00,  1.7843075e+01,  9.4569902e+00, 7.8301370e-01]
variances = [2.0686106e-01, 1.2035955e+01, 2.8855536e+02, 2.5339364e+05, 1.6990326e-01]

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = build_lasernet_functional(means=means, variances=variances)
schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 150, 0.99)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
model.compile(optimizer=optimizer, loss={'classes': ClassLoss()})

# print(logs + '/images')
file_writer = tf.summary.create_file_writer(logs + '/images')
repeated_ds = final_ds.take(1).cache()
def evaluate_images(epoch, logs):
    for data, label in repeated_ds:
        test_pred_raw = model.predict(data)
        test_pred = tf.keras.layers.Softmax()(test_pred_raw)[..., 1:]
        test_label = tf.int32.max * label['classes'][..., 0:1]
        with file_writer.as_default():
            tf.summary.image("Prediction", test_pred, step=epoch, max_outputs=4)
            tf.summary.image("Label", test_label, step=epoch, max_outputs=4)
        break
            # Prints false if images are not written because epoch length not divisible by update_freq

model.fit(final_ds, epochs=20, steps_per_epoch=2470, callbacks = [tboard_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_images), model_checkpoint_callback])