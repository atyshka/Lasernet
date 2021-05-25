import tensorflow as tf
from tensorflow import keras
from .transforms import PredictionTransform, BoxToCorners
from .dla import FeatureAggregator, FeatureExtractor

class LaserNet(keras.Model):
    def __init__(self, **kwargs):
        super(LaserNet, self).__init__(**kwargs)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        self.block_1a = FeatureExtractor(64, downsample=False, reshape=True, dtype=policy)
        self.block_1b = FeatureAggregator(64, dtype=policy)
        self.block_2a = FeatureExtractor(64, dtype=policy)
        self.block_1c = FeatureAggregator(64, dtype=policy)
        self.block_2b = FeatureAggregator(64, dtype=policy)
        self.block_3a = FeatureExtractor(64, dtype=policy)
        self.predict = PredictionTransform(1, [1])
        self.corners = BoxToCorners()

    @tf.function
    def call(self, inputs, training=False):
        (inputs, xy) = inputs
        extract_1 = self.block_1a(inputs, training=training)
        extract_2 = self.block_2a(extract_1, training=training)
        extract_3 = self.block_3a(extract_2, training=training)
        aggregate_1 = self.block_1b(extract_1, extract_2, training=training)
        aggregate_2 = self.block_2b(extract_2, extract_3, training=training)
        raw_output = tf.cast(self.block_1c(aggregate_1, aggregate_2, training=training), tf.float32)
        classes, centers, alphas, log_stddevs = self.predict(raw_output, training=training)
        azimuth = inputs[:, :, :, 2]
        corners = self.corners(centers, xy, azimuth)
        box_params = tf.concat([corners, tf.expand_dims(alphas, -1), tf.expand_dims(log_stddevs, -1)], -1)
        box_shape = box_params.get_shape()
        return tf.concat([classes, tf.reshape(box_params, [box_shape[0], box_shape[1], box_shape[2], -1])], axis=-1)

def build_lasernet_functional(batch_size=4,means=0, variances=1, policy='mixed_float16'):
    policy = 'float32'#tf.keras.mixed_precision.Policy(policy)
    inputs = keras.Input(shape=(64, 2648, 5), name="input_laser")
    xyz = keras.Input(shape=(64, 2648, 3), name="input_xyz")
    xy = xyz[..., 0:2]
    inputs_norm = keras.layers.experimental.preprocessing.Normalization(mean=means, variance=variances)(inputs)
    extract_1 = FeatureExtractor(64, downsample=False, reshape=True, dtype=policy, name="extract_1")(inputs_norm)
    extract_2 = FeatureExtractor(64, dtype=policy, name="extract_2")(extract_1)
    extract_3 = FeatureExtractor(128, dtype=policy, name="extract_3")(extract_2)
    aggregate_1 = FeatureAggregator(64, dtype=policy, name="aggregate_1")(extract_1, extract_2)
    aggregate_2 = FeatureAggregator(128, dtype=policy, name="aggregate_2")(extract_2, extract_3)
    raw_output = tf.cast(FeatureAggregator(128, dtype=policy, name="aggregate_3")(aggregate_1, aggregate_2), tf.float32)
    classes, centers, alphas, log_stddevs = PredictionTransform(1, [1])(raw_output)
    classes = keras.layers.Lambda(lambda x: x, name="classes")(classes)
    azimuth = inputs[..., 2]
    corners = BoxToCorners()(centers, xy, azimuth)
    box_params = tf.concat([corners, tf.expand_dims(alphas, -1), tf.expand_dims(log_stddevs, -1)], -1)
    box_shape = tf.shape(box_params)
    box_reshaped = tf.reshape(box_params, [box_shape[0], box_shape[1], box_shape[2], -1])
    box_reshaped = keras.layers.Lambda(lambda x: x, name="boxes")(box_reshaped)
    return keras.Model(inputs=[inputs, xyz], outputs=[classes, box_reshaped])