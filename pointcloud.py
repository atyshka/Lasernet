import tensorflow as tf
import math
import numpy as np
from tensorboard.plugins.mesh import summary_v2

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils, box_utils

file_ds = tf.data.Dataset.list_files('gs://waymo_open_dataset_v_1_2_0_individual_files/training/*.tfrecord')
record_ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)

def parse_frame(data):
    frame = open_dataset.Frame()
    frame.ParseFromString(data.numpy())
    top_image = frame_utils.parse_range_image_and_camera_projection(frame)[0][open_dataset.LaserName.TOP][0]
    range_image_tensor = tf.reshape(tf.convert_to_tensor(top_image.data), top_image.shape.dims)
    frame.context.laser_calibrations.sort(key=lambda laser: laser.name)
    c = frame.context.laser_calibrations[open_dataset.LaserName.TOP-1]
    # tf.print(c)
    extrinsic = tf.reshape(tf.convert_to_tensor(c.extrinsic.transform), [4,4])
    beam_inclinations = tf.convert_to_tensor(c.beam_inclinations)
    vehicle_labels = []
    for label in frame.laser_labels:
        if label.type == label.TYPE_VEHICLE:
            vehicle_labels.append([label.box.center_x, label.box.center_y, label.box.length, label.box.width, label.box.heading])
    num_labels = len(vehicle_labels)
    vehicle_labels = tf.convert_to_tensor(vehicle_labels)
    return range_image_tensor, extrinsic, beam_inclinations, vehicle_labels

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

def fill_boxes(input, nlz, indices, labels):
    gathered = tf.gather(labels, indices, axis=1, batch_dims=1)
    dense_boxes = corners_from_centers(gathered)
    classes = tf.where(tf.greater_equal(indices, 1), 1, 0)
    classes = classes[..., 0:2648]
    return input, {'classes': tf.stack([classes, tf.cast(input['input_laser'][..., 4], tf.int32), tf.cast(nlz, tf.int32)], -1)}#, "boxes": tf.concat([dense_boxes, tf.expand_dims(tf.cast(indices, tf.float32), -1)], -1)}

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

# tensor_ds = tf.data.experimental.sample_from_datasets(datasets)
tensor_ds = record_ds.map(lambda x: tf.py_function(func=parse_frame, inp=[x], Tout=(tf.float32, tf.float32, tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
tensor_ds = tensor_ds.map(shape, num_parallel_calls=32)
tensor_ds = tensor_ds.map(transform, num_parallel_calls=32)

data = next(iter(tensor_ds))
points = tf.reshape(data['input_xyz'], [-1, 3]).numpy()

cloud_writer = tf.summary.create_file_writer('cloud_logs')
with cloud_writer.as_default():
    az = data['input_laser'][..., 0]
    hue = (az + math.pi) / (4*math.pi)
    saturation = tf.ones_like(hue);
    value = tf.ones_like(hue)
    rgb = tf.image.hsv_to_rgb(tf.stack([hue, saturation, value], -1)) * 255
    summary_v2.mesh('mesh', tf.reshape(data['input_xyz'], [1, -1, 3]), colors=tf.reshape(rgb, [1, -1, 3]), step=0, config_dict={"material": {'cls': 'PointsMaterial','size': 0.1}})
# print('points created')
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# print('drawing')
# o3d.visualization.draw_geometries([pcd])