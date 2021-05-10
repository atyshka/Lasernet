import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResnetBlock(layers.Layer):
  def __init__(self, num_filters, kernel_size=(3, 3), strides=(1, 1), reshape=False, name=None, dtype=None):
    super(ResnetBlock, self).__init__(name=name, dtype=dtype)

    self.conv1 = layers.Conv2D(num_filters, kernel_size, strides=strides, use_bias=False, padding='same', dtype=dtype)
    self.bn1 = layers.BatchNormalization(dtype=dtype)

    self.conv2 = layers.Conv2D(num_filters, kernel_size, use_bias=False, padding='same', dtype=dtype)
    self.bn2 = layers.BatchNormalization(dtype=dtype)

    if reshape or strides != (1, 1):
        # Need to match skip connection dimensions to new dimensions
        self.conv3 = layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=strides, use_bias=False, dtype=dtype)
        self.bn3 = layers.BatchNormalization(dtype=dtype)
        self.skip_conn = lambda x, **kwargs: self.bn3(self.conv3(x))
        # self.skip_conn.add()
        # self.skip_conn.add()
    else:
        # Do nothing
        self.skip_conn = lambda x, **kwargs: x

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    residual = self.skip_conn(input_tensor, training=training)
    x += residual
    return tf.nn.relu(x)

class FeatureExtractor(layers.Layer):
    def __init__(self, num_filters, num_blocks=6, downsample=True, reshape=False, dtype=None, name='FeatureExtractor'):
        super(FeatureExtractor, self).__init__(name=name, dtype=dtype)

        self.sequence = keras.Sequential()
        # Downsample by 2 along horizontal
        if downsample:
            self.sequence.add(ResnetBlock(num_filters, strides=(1, 2), name="Downsample", reshape=reshape, dtype=dtype))
        for i in range(num_blocks):
            self.sequence.add(ResnetBlock(num_filters, name="Resnet_%i" % i, reshape=reshape, dtype=dtype))

    def call(self, input_tensor, training=False):
        return self.sequence(input_tensor, training=training)

class FeatureAggregator(layers.Layer):
    def __init__(self, num_filters, dtype=None, name='FeatureAggregator'):
        super(FeatureAggregator, self).__init__(name=name, dtype=dtype)

        self.upsample = layers.Conv2DTranspose(num_filters, (3, 3), (1,2), padding='same', dtype=dtype)
        self.bn = layers.BatchNormalization(dtype=dtype)
        self.concat = layers.Concatenate(dtype=dtype)
        self.block1 = ResnetBlock(num_filters, reshape=True, dtype=dtype)
        self.block2 = ResnetBlock(num_filters, dtype=dtype)

    def call(self, fine_input, coarse_input, training=False):
        x = self.upsample(coarse_input)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        y = self.concat([fine_input, x])
        y = self.block1(y, training=training)
        return self.block2(y, training=training)