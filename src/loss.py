import tensorflow as tf
from tensorflow.keras import losses

class LasernetLoss(losses.Loss):
    def __init__(self, object_classes=3, mixture_components=[1, 1, 1]):
        super(LasernetLoss, self).__init__(name='output_transform')
        self.object_classes = object_classes
        self.mixture_components = mixture_components

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, gamma=2):
        shape = y_pred.get_shape()
        num_pixels = shape[1] * shape[2]
        num_classes = self.object_classes + 1
        num_components = sum(self.mixture_components)
        # B, W, H, Classes
        pred_class, pred_box = tf.split(y_pred, [num_classes, 10 * num_components], axis=3)
        
        truth_class, truth_corners, truth_object_id = tf.split(y_true, [1, 8, 1], axis=-1)
        truth_class_onehot = tf.one_hot(truth_class, num_classes)
        categorical_loss = losses.categorical_crossentropy(truth_class_onehot, pred_class)
        pt = tf.exp(-1 * categorical_loss)
        focal_loss = tf.pow(1 - pt, gamma) * categorical_loss
        class_loss = tf.reduce_sum(focal_loss) / num_pixels

        # Reshape box params to a ragged tensor
        # B x H x W x (C*k) X 10
        # Flatten to BHWC*K x 10
        flattened = tf.reshape(pred_box, [pred_box.get_shape().num_elements() / 10, 10])
        # Expand to BHWC x None x 10
        s1 = tf.RaggedTensor.from_row_lengths(flattened, self.mixture_components)
        # Expand to BHW x C x None x 10
        s2 = tf.RaggedTensor.from_uniform_row_length(s1, shape[2])
        # Expand to BH x W x C x None x 10
        s3 = tf.RaggedTensor.from_uniform_row_length(s2, shape[1])
        # Expand to B x H x W x C x None x 10
        ragged_box_pred = tf.RaggedTensor.from_uniform_row_length(s3, shape[0])
        # B x H x W x None x 10
        class_matched_box_pred = tf.gather(ragged_box_pred, truth_class, axis=3, batch_dims=3)
        pred_corners = class_matched_box_pred[:, :, :, :, :8]
        corner_norm: tf.RaggedTensor = tf.math.sqrt(tf.reduce_sum(tf.math.square(pred_corners - truth_corners), axis=-1))
        # To match to the mixture component with lowest norm, we need to convert ragged tensor to regular
        # Fill empty values with max possible value so they don't show up in min
        # B x H x W x 1
        min_indices = tf.math.argmin(corner_norm.to_tensor(default_value=corner_norm.dtype.max))
        # B x H x W x 10
        matched_predictions = tf.gather(class_matched_box_pred, min_indices, axis=3, batch_dims=3)
        matched_corners, _, matched_log_stddev = tf.split(matched_predictions, [8, 1, 1], axis=-1)
        box_loss = (tf.reduce_sum(tf.abs(matched_corners - truth_corners), axis=-1) / tf.exp(matched_log_stddev))  + matched_log_stddev
        mixture_logits = class_matched_box_pred[:, :, :, :, 8]
        # B x H x W x max(k)
        mixture_logits_tensor = mixture_logits.to_tensor(default_value=mixture_logits.dtype.min)
        truth_mixture = tf.one_hot(min_indices, mixture_logits_tensor.get_shape()[-1])
        mixture_loss = tf.keras.losses.categorical_crossentropy(truth_mixture, mixture_logits_tensor, from_logits=True)
        regression_loss = box_loss + 0.25 * mixture_loss
        flattened_regression_loss = tf.reshape(regression_loss, [regression_loss.get_shape().num_elements, 1])
        flattened_object_ids = tf.reshape(truth_object_id, [truth_object_id.get_shape().num_elements, 1])
        num_objects = tf.math.reduce_max(flattened_object_ids)
        # Find the mean loss for each object. Points not on an object are assigned object id 0, and sliced off before computing a final mean loss for all objects
        # Note that we use this weighting technique so that objects with many points do not outweigh those with few
        weighted_regression_loss = tf.reduce_mean(tf.math.unsorted_segment_mean(flattened_regression_loss, flattened_object_ids, num_objects + 1)[1:])
        
        total_loss = class_loss + weighted_regression_loss
        return total_loss
