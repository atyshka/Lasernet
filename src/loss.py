import tensorflow as tf
from tensorflow.keras import losses

class LaserNetLoss(losses.Loss):
    def __init__(self, object_classes=1, mixture_components=[1]):
        super(LaserNetLoss, self).__init__(name='loss')
        self.object_classes = object_classes
        self.mixture_components = mixture_components

    def call(self, y_true: tf.Tensor, y_pred: (tf.Tensor, tf.Tensor), gamma=2):

        shape = y_pred.get_shape()
        num_pixels = shape[0] * shape[1] * shape[2]
        num_classes = self.object_classes + 1
        num_components = sum(self.mixture_components)
        # B, W, H, Classes
        # (pred_class, pred_box_raw) = y_pred
        pred_class = y_pred[..., :num_classes]
        pred_box_raw = y_pred[..., num_classes:]
        print(pred_box_raw.shape)
        pred_box = tf.pad(pred_box_raw, [[0, 0], [0, 0], [0, 0], [10, 0]])
        print(type(y_true))
        print(pred_box.shape)
        truth_class, truth_corners, truth_object_id = tf.split(y_true, [1, 8, 1], axis=-1)
        truth_class = tf.squeeze(tf.cast(truth_class, tf.int32), -1)
        truth_object_id = tf.squeeze(tf.cast(truth_object_id, tf.int32), -1)
        # (truth_class, truth_corners, truth_object_id) = y_true
        truth_class_onehot = tf.one_hot(truth_class, num_classes)
        categorical_loss = losses.categorical_crossentropy(truth_class_onehot, pred_class)
        pt = tf.exp(-1 * categorical_loss)
        focal_loss = tf.pow(1 - pt, gamma) * categorical_loss
        class_loss = tf.reduce_sum(focal_loss) / num_pixels

        # Reshape box params to a ragged tensor
        # B x H x W x (C*k) X 10
        # Flatten to BHWC*K x 10
        # flattened = tf.reshape(pred_box, [-1, 10])
        # # Expand to BHWC x None x 10
        # print(flattened.shape)
        # s1 = tf.RaggedTensor.from_row_lengths(flattened, tf.tile(self.mixture_components, [self.object_classes * num_pixels]))
        # print(s1.shape)
        # # Expand to BHW x C x None x 10
        # s2 = tf.RaggedTensor.from_uniform_row_length(s1, shape[2])
        # print(s2.shape)
        # # Expand to BH x W x C x None x 10
        # s3 = tf.RaggedTensor.from_uniform_row_length(s2, shape[1])
        # # Expand to B x H x W x C x None x 10
        # print(s3.shape)
        # ragged_box_pred = s3 #tf.RaggedTensor.from_uniform_row_length(s3, shape[0])
        # print(ragged_box_pred.shape)
        # Fill the background class with zeroes
        car_pred = tf.reshape(pred_box, [pred_box.shape[0], pred_box.shape[1], pred_box.shape[2], -1, 10])

        # B x H x W x None x 10
        class_matched_predictions = tf.gather(car_pred, truth_class, axis=3, batch_dims=3)
        # pred_corners = class_matched_box_pred[:, :, :, :8]
        # corner_norm: tf.RaggedTensor = tf.math.sqrt(tf.reduce_sum(tf.math.square(pred_corners - truth_corners), axis=-1))
        # To match to the mixture component with lowest norm, we need to convert ragged tensor to regular
        # Fill empty values with max possible value so they don't show up in min
        # B x H x W x 1
        # min_indices = tf.math.argmin(corner_norm.to_tensor(default_value=corner_norm.dtype.max))
        # B x H x W x 10
        # matched_predictions = tf.gather(class_matched_box_pred, min_indices, axis=3, batch_dims=3)
        matched_corners, _, matched_log_stddev = tf.split(class_matched_predictions, [8, 1, 1], axis=-1)
        box_loss = (tf.reduce_sum(tf.abs(matched_corners - truth_corners), axis=-1) / tf.squeeze(tf.exp(tf.clip_by_value(matched_log_stddev, -20, 20)), axis=-1))  + tf.squeeze(matched_log_stddev, axis=-1)
        # mixture_logits = class_matched_predictions[:, :, :, :, 8]
        # B x H x W x max(k)
        # mixture_logits_tensor = mixture_logits.to_tensor(default_value=mixture_logits.dtype.min)
        # truth_mixture = tf.one_hot(min_indices, mixture_logits_tensor.get_shape()[-1])
        # mixture_loss = tf.keras.losses.categorical_crossentropy(truth_mixture, mixture_logits_tensor, from_logits=True)
        regression_loss = box_loss #+ 0.25 * mixture_loss
        flattened_regression_loss = tf.reshape(regression_loss, [-1, 1])
        flattened_object_ids = tf.reshape(truth_object_id, [-1, 1])
        num_objects = tf.math.reduce_max(flattened_object_ids)
        # Find the mean loss for each object. Points not on an object are assigned object id 0, and sliced off before computing a final mean loss for all objects
        # Note that we use this weighting technique so that objects with many points do not outweigh those with few
        weighted_regression_loss = tf.reduce_mean(tf.math.unsorted_segment_mean(flattened_regression_loss, flattened_object_ids, num_objects + 1)[1:])
        
        total_loss = class_loss #+ weighted_regression_loss
        return total_loss

class ClassLoss(losses.Loss):
    def __init__(self, reduction=None, name='class_loss', object_classes=1, mixture_components=[1]):
        super(ClassLoss, self).__init__(name=name)
        self.object_classes = object_classes
        self.mixture_components = mixture_components

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, gamma=2):

        shape = y_pred.get_shape()
        num_pixels = shape[0] * shape[1] * shape[2]
        num_classes = self.object_classes + 1
        num_components = sum(self.mixture_components)
        # B, W, H, Classes
        # (pred_class, pred_box_raw) = y_pred
        pred_class = y_pred
        truth_class = y_true
        # (truth_class, truth_corners, truth_object_id) = y_true
        truth_class_onehot = tf.one_hot(truth_class, num_classes)
        categorical_loss = losses.categorical_crossentropy(truth_class_onehot, pred_class, from_logits=True)
        pt = tf.exp(-1 * categorical_loss)
        focal_loss = tf.pow(1 - pt, gamma) * categorical_loss
        class_loss = tf.reduce_sum(focal_loss) / num_pixels
        return class_loss