import tensorflow as tf

from Structured.utils.operations import max_pool

class ROIPooling:
    def __init__(self, config, roi_proposals, conv_feature_map, img_shape):
        assert len(config.roi_pooling_size) == 2

        self.config                 = config

        self.roi_pooling_size       = self.config.roi_pooling_size
        self.roi_pooling_padding    = self.config.roi_pooling_padding

        self.roi_pooled_features    = self.roi_crop(roi_proposals, conv_feature_map, img_shape)


    def get_bboxes(self, roi_proposals, img_shape):
        """
        Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
        in TensorFlow's order (y1, x1, y2, x2).
        Args:
            roi_proposals: A Tensor with the bounding boxes of shape
                (total_proposals, 5), where the values for each proposal are
                (x_min, y_min, x_max, y_max).
            img_shape: A Tensor with the shape of the image (height, width).
        Returns:
            bboxes: A Tensor with normalized bounding boxes in TensorFlow's
                format order. Its should is (total_proposals, 4).
        """
        with tf.name_scope('get_bboxes'):
            img_shape = tf.cast(img_shape, tf.float32)

            x1, y1, x2, y2 = tf.unstack(
                roi_proposals, axis=1
            )

            x1 = x1 / img_shape[1]
            y1 = y1 / img_shape[0]
            x2 = x2 / img_shape[1]
            y2 = y2 / img_shape[0]

            bboxes = tf.stack([y1, x1, y2, x2], axis=1)

            return bboxes

    def roi_crop(self, roi_proposals, conv_feature_map, img_shape):
        # Get normalized bounding boxes.
        bboxes          = self.get_bboxes(roi_proposals, img_shape)
        # Generate fake batch ids
        bboxes_shape    = tf.shape(bboxes)
        batch_ids       = tf.zeros((bboxes_shape[0],), dtype=tf.int32)

        # Apply crop and resize with extracting a crop double the desired size.

        pooled_width, pooled_height = self.roi_pooling_size

        crops = tf.image.crop_and_resize(
            conv_feature_map, bboxes, batch_ids,
            [pooled_width * 2, pooled_height * 2], name="crops"
        )

        # Applies max pool with [2,2] kernel to reduce the crops to half the
        # size, and thus having the desired output.
        prediction = max_pool(
            crops, 2, 2, 2, 2, name="roi_max_pool", padding=self.roi_pooling_padding
        )

        return prediction

