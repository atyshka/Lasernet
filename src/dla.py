import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ResnetBlock(layers.Layer):
  def __init__(self, num_filters, kernel_size=(3, 3), strides=(1, 1), reshape=False):
    super(ResnetBlock, self).__init__(name='ResnetBlock')

    self.conv1 = layers.Conv2D(num_filters, kernel_size, strides=strides, use_bias=False, padding='same')
    self.bn1 = layers.BatchNormalization()

    self.conv2 = layers.Conv2D(num_filters, kernel_size, use_bias=False, padding='same')
    self.bn2 = layers.BatchNormalization()

    if reshape or strides != (1, 1):
        # Need to match skip connection dimensions to new dimensions
        self.skip_conn = keras.Sequential()
        self.skip_conn.add(layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=strides, use_bias=False))
        self.skip_conn.add(layers.BatchNormalization())
    else:
        # Do nothing
        self.skip_conn = lambda x: x

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
    def __init__(self, num_filters, num_blocks=6, downsample=True):
        super(FeatureExtractor, self).__init__(name='FeatureExtractor')

        self.sequence = keras.Sequential()
        # Downsample by 2 along horizontal
        if downsample:
            self.sequence.add(ResnetBlock(num_filters, strides=(2, 1)))
        for _ in range(num_blocks):
            self.sequence.add(ResnetBlock(num_filters))

    def call(self, input_tensor, training=False):
        return self.sequence(input_tensor, training=training)

class FeatureAggregator(layers.Layer):
    def __init__(self, num_filters, downsample=True):
        super(FeatureAggregator, self).__init__(name='FeatureAggregator')

        self.upsample = layers.Conv2DTranspose(num_filters, (3, 3), padding='same')
        self.bn = layers.BatchNormalization()
        self.concat = layers.Concatenate()
        self.block1 = ResnetBlock(num_filters, reshape=True)
        self.block2 = ResnetBlock(num_filters)

    def call(self, fine_input, coarse_input, training=False):
        x = self.upsample(coarse_input)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        y = self.concat(fine_input, x)
        y = self.block1(y, training=training)
        return self.block2(y)

class OutputTransform(layers.Layer):
    def __init__(self, num_classes, mixture_components=None):
        super(OutputTransform, self).__init__(name="OutputTransform")

        # Class Probabilities plus background
        num_filters = num_classes + 1
        for i in range(num_classes):
            k = mixture_components[i] if mixture_components is not None else 1
            # 8 components: dx, dy, wx, wy, l, w, s, alpha
            num_filters += (k * 8)
        self.conv = layers.Conv2D(num_filters, (1, 1), use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)