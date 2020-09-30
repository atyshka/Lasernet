import tensorflow as tf
from tensorflow import keras
from transforms import PredictionTransform, BoxToCorners
from dla import FeatureAggregator, FeatureExtractor

class LaserNet(keras.Model):
    def __init__(self):
        super(LaserNet, self).__init__()
        self.block_1a = FeatureExtractor(64, downsample=False)
        self.block_1b = FeatureAggregator(64)
        self.block_2a = FeatureExtractor(64)
        self.block_1c = FeatureAggregator(64)
        self.block_2b = FeatureAggregator(64)
        self.block_3a = FeatureExtractor(64)
        self.predict = PredictionTransform(1, [1])
        self.corners = BoxToCorners()

    def call(self, inputs, xy, training=False):
        extract_1 = self.block_1a(inputs, training=training)
        extract_2 = self.block_2a(extract_1, training=training)
        extract_3 = self.block_3a(extract_2, training=training)
        aggregate_1 = self.block_1b(extract_1, extract_2, training=training)
        aggregate_2 = self.block_2b(extract_2, extract_3, training=training)
        raw_output = self.block_1c(aggregate_1, aggregate_2, training=training)
        classes, centers, alphas, log_stddevs = self.predict(raw_output, training=training)
        azimuth = inputs[:, :, :, 2]
        corners = self.corners(centers, xy, azimuth)
        return classes, corners, alphas, log_stddevs
