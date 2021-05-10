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

file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/training/*.tfrecord').shuffle(100)
# file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord')
record_ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

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
    return range_image_tensor, extrinsic, beam_inclinations, vehicle_labels

tensor_ds = map_py_function_to_dataset(record_ds, parse_frame_parallel, 32, (tf.float32, tf.float32, tf.float32, tf.float32)).shuffle(2048)

def shard_func(w, x, y, z):
    return tf.random.uniform([1], minval=0, maxval=20, dtype=tf.int64)

tf.data.experimental.save(tensor_ds, '/home/alex/lasernet-waymo/compressed_ds', compression='GZIP', shard_func=shard_func)
