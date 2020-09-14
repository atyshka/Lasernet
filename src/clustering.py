import tensorflow as tf
from timeit import timeit
from tensorflow import keras, sparse
from tensorflow.keras import layers

class ProjectBEV(layers.Layer):
    def __init__(self, min_x, max_x, min_y, max_y, num_classes, mixture_components=None, resolution = 0.5):
        super(ProjectBEV, self).__init__(name='ProjectBEV')

        # We pad the grid by one on each side for the presence and position tensors
        # Items that fall outside region bounds will be clipped to padding region
        # For actual mean shift, this padding will be sliced off 
        self.length = int((max_x - min_x) / resolution) + 2
        self.width = int((max_y - min_y) / resolution) + 2
        self.scale_factor = 1.0 / resolution
        self.x_offset = (-1 * min_x * self.scale_factor) + 1
        self.y_offset = (-1 * min_y * self.scale_factor) + 1
        self.num_channels = 0
        if mixture_components is not None:
            for i in range (num_classes):
                self.num_channels += mixture_components[i]
        else:
            self.num_channels = num_classes

    def build(self, input_shape: tf.TensorShape):
        # Input Size B x Hi x Wi x C x 2
        # Reorder to B x C x Hi x Wi x 2
        self.reordered_input_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2], input_shape[4]]
        presence_mask = tf.fill(self.reordered_input_shape[:-1], True)
        presence_indices = tf.where(presence_mask)
        self.presence_b = tf.expand_dims(presence_indices[:, 0], axis=-1)
        self.presence_chw = presence_indices[:, 1:]

        position_mask = tf.fill(self.reordered_input_shape, True)
        position_indices = tf.where(position_mask)
        self.position_b = tf.expand_dims(position_indices[:, 0], axis=-1)
        self.position_chwd = position_indices[:, 1:]
        # Size B x Hi x Wi x C x 2
        self.input_size = tf.cast(input_shape.num_elements(), tf.int32) / 2
        self.xy_offset = tf.repeat([[self.x_offset, self.y_offset]], repeats=[self.input_size], axis=0)
        self.presence_shape = [input_shape[0], self.length, self.width, self.num_channels, input_shape[1], input_shape[2]]
        self.presence_vals = tf.ones([self.input_size], dtype=tf.int32)
        self.position_shape = [input_shape[0], self.length, self.width, self.num_channels, input_shape[1], input_shape[2], 2]

    def call(self, input_tensor: tf.Tensor):
        """Takes a dense range image and projects it to a sparse 2d grid

        Args:
            input_tensor (tf.Tensor): Input range image. Expected shape is B x H x W x CK x 2, 
                where CK is each mixture model component for each class. 
                Innermost values should be relevant xy pair
            training (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # Input Size B x Hi x Wi x C x 2
        # Reorder to B x C x Hi x Wi x 2
        reordered_input = tf.transpose(input_tensor, perm=[0, 3, 1, 2, 4])
        # Reshape to B*C*Hi*Wi x 2
        flattened_xy = tf.reshape(reordered_input, [self.input_size, 2])
        scaled_xy = tf.scalar_mul(self.scale_factor, flattened_xy) + self.xy_offset
        quantized_xy = tf.cast(scaled_xy, tf.int64)
        # [B, X, Y, C, Hi, Wi]
        presence_indices = tf.concat([self.presence_b, quantized_xy, self.presence_chw], -1)
        presence_tensor = sparse.SparseTensor(indices=presence_indices, values=self.presence_vals, dense_shape=self.presence_shape)\
        
        position_vals = tf.reshape(scaled_xy, [self.input_size * 2])
        position_indices = tf.concat([self.position_b, tf.repeat(quantized_xy, 2, 0), self.position_chwd], -1)
        position_tensor = sparse.SparseTensor(indices=position_indices, values=position_vals, dense_shape=self.position_shape)
        
        return position_tensor, position_tensor




        
if __name__ == "__main__":
    input_ones = tf.ones([5, 2048, 64, 3, 2])
    bev = ProjectBEV(0, 70, -70, 70, 3)
    print(timeit(lambda: bev(input_ones), number=10))
        