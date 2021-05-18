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


datasets = []
for i in range(0, 20):
    ds = tf.data.experimental.load('/home/alex/dataset-drive/ds_sharded/shard_%d'%i, (
                            tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                            tf.TensorSpec(shape=[16], dtype=tf.float32), 
                            tf.TensorSpec(shape=[64], dtype=tf.float32), 
                            tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')
    datasets.append(ds)

tensor_ds = tf.data.experimental.sample_from_datasets(datasets)
tensor_ds = tensor_ds.map(shape, num_parallel_calls=32)
tensor_ds = tensor_ds.map(transform, num_parallel_calls=32)
tensor_ds = tensor_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=128, drop_remainder=True))
final_ds = tensor_ds.map(fill_boxes, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)

norm_1 = keras.layers.experimental.preprocessing.Normalization()
input_ds = final_ds.map(lambda x, y: x['input_laser'])
norm_1.adapt(input_ds)
print(norm_1.mean)
print(norm_1.variance)