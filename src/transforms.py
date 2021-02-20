import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PredictionTransform(layers.Layer):
    def __init__(self, object_classes, mixture_components=None):
        super(PredictionTransform, self).__init__(name="PredictionTransform")

        # Class Probabilities plus background
        self.num_classes = object_classes + 1
        self.num_components = sum(mixture_components) if mixture_components is not None else object_classes
        self.conv = layers.Conv2D(self.num_classes + (8 * self.num_components), (1, 1), use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, input, training=False):
        x = self.conv(input)
        pred_class, pred_boxes = tf.split(x, [self.num_classes, self.num_components * 8], -1)
        pred_centers, pred_alpha, pred_log_stddev = tf.split(pred_boxes, [6 * self.num_components, self.num_components, self.num_components], -1)
        return pred_class, pred_centers, pred_alpha, pred_log_stddev

class BoxToCorners(layers.Layer):
    def __init__(self):
        super(BoxToCorners, self).__init__(name="BoxToCorners")

    def call(self, center_params, xy, azimuth):
        # B H W CK 4
        rotation_mat_flat = tf.stack([tf.math.cos(azimuth), -1 * tf.math.sin(azimuth), 
                                        tf.math.sin(azimuth), tf.math.cos(azimuth)], axis=-1)
        # B H W CK 2 2
        rotation_mat = tf.reshape(rotation_mat_flat, tf.concat([rotation_mat_flat.get_shape(), [2, 2]], 0))
        input_shape = center_params.get_shape()
        # B H W CK 6
        centers_grouped = tf.reshape(center_params, [input_shape[0], input_shape[1], input_shape[2], -1, 6])
        xy_offsets, sin, cos, length, width = tf.split(centers_grouped, [2, 1, 1, 1, 1], -1)
        # B H W CK 1 2
        new_centers = tf.expand_dims(xy, -2) + rotation_mat @ tf.expand_dims(xy_offsets, -2)
        orientation = azimuth + tf.math.atan2(sin, cos)
        orientation_mat_flat = tf.stack([tf.math.cos(orientation), -1 * tf.math.sin(orientation), 
                                        tf.math.sin(orientation), tf.math.cos(orientation)], axis=-1)
        corner1 = new_centers + 0.5 * (orientation_mat_flat @ tf.stack([length, width], axis=-1))
        corner2 = new_centers + 0.5 * (orientation_mat_flat @ tf.stack([length, -1 * width], axis=-1))
        corner3 = new_centers + 0.5 * (orientation_mat_flat @ tf.stack([-1 * length, -1 * width], axis=-1))
        corner4 = new_centers + 0.5 * (orientation_mat_flat @ tf.stack([-1 * length, width], axis=-1))
        # B H W CK 1 8
        concat = tf.concat([corner1, corner2, corner3, corner4], -1)
        # B H W CK 8
        return tf.squeeze(concat, 4)
