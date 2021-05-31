import tensorflow as tf
import numpy as np

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

file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/validation/*.tfrecord')
# record_ds = file_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
#     cycle_length=len(file_ds), num_parallel_calls=4,
#     deterministic=False)
record_ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

def parse_frame_parallel(data):
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())
    top_image, _, top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    top_image = top_image[open_dataset.LaserName.TOP][0]
    range_image_tensor = tf.convert_to_tensor(top_image.data)
    range_image_top_pose_tensor = tf.convert_to_tensor(top_pose.data)
    frame.context.laser_calibrations.sort(key=lambda laser: laser.name)
    c = frame.context.laser_calibrations[open_dataset.LaserName.TOP-1]
    extrinsic = tf.convert_to_tensor(c.extrinsic.transform)
    beam_inclinations = tf.convert_to_tensor(c.beam_inclinations)
    vehicle_labels = []
    pedestrian_labels = []
    cyclist_labels = []
    for label in frame.laser_labels:
        if label.type == label.TYPE_VEHICLE:
            vehicle_labels.append([label.box.center_x, label.box.center_y, label.box.length, label.box.width, label.box.heading])
        elif label.type == label.TYPE_CYCLIST:
            cyclist_labels.append([label.box.center_x, label.box.center_y, label.box.length, label.box.width, label.box.heading])
        elif label.type == label.TYPE_PEDESTRIAN:
            pedestrian_labels.append([label.box.center_x, label.box.center_y, label.box.length, label.box.width, label.box.heading])
    vehicle_labels = tf.convert_to_tensor(vehicle_labels)
    pedestrian_labels = tf.convert_to_tensor(pedestrian_labels)
    cyclist_labels = tf.convert_to_tensor(cyclist_labels)
    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    return range_image_tensor, extrinsic, beam_inclinations, vehicle_labels, pedestrian_labels, cyclist_labels, range_image_top_pose_tensor, frame_pose

def top_pose(range_image_tensor, extrinsic, beam_inclinations, vehicle_labels, pedestrian_labels, cyclist_labels, range_image_top_pose_tensor, frame_pose):
    range_image_tensor = tf.reshape(range_image_tensor, [64, 2650, 4])
    range_image_top_pose_tensor = tf.reshape(range_image_top_pose_tensor, [64, 2650, 6])
    extrinsic = tf.reshape(extrinsic, [4,4])
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    pixel_pose_local = range_image_top_pose_tensor
    pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
    frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    
    return range_image_tensor, extrinsic, beam_inclinations, vehicle_labels, pedestrian_labels, cyclist_labels, range_image_cartesian

tensor_ds = map_py_function_to_dataset(record_ds, parse_frame_parallel, 32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float64))
tensor_ds = tensor_ds.map(top_pose, tf.data.AUTOTUNE)

def shard_func(w, x, y, z, a, b, c):
    return tf.random.uniform([1], minval=0, maxval=20, dtype=tf.int64)

print(tensor_ds.element_spec)
tf.data.experimental.save(tensor_ds, '/home/alex/full_validation', compression='GZIP', shard_func=shard_func)
# for i in range(20):
#     shard = tensor_ds.take(7904)
#     shard = shard.shuffle(7904)
#     tf.data.experimental.save(shard, '/home/alex/dataset-drive/ds_sharded_final/shard_%d'%i, compression='GZIP', shard_func=lambda w,x,y,z: tf.constant(i, dtype=tf.int64))
#     tensor_ds = tensor_ds.skip(7904)