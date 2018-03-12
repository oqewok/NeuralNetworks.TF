import tensorflow as tf
import numpy as np


class RPN:
    """RPN - Region Proposal Network.
            Given an image (as feature map) and a fixed set of anchors, the RPN
            will learn weights to adjust those anchors so they better look like the
            ground truth objects, as well as scoring them by "objectness" (ie. how
            likely they are to be an object vs background).
            The final result will be a set of rectangular boxes ("proposals"),
            each associated with an objectness score.
            Note: this module can be used independently of Faster R-CNN.

            Дано изображение (представленное в виде feature maps) и фиксированных набор "якорей", RPN
            будет обучать веса для регулировки этих "якорей", чтобы они лучше соответствовали
            ground truth объектам, также оценивая их результат на "предметность (objectness)"
            (к чему их относить - к объекту или фону).
            Конечным результатом является набор прямоугольных областей ("предположений (proposals))",
            каждая из которых ассоциирована со своим результатом на "предметность (objectness)".

            Замечание: данный модуль может быть использован независимо от Faster R-CNN.
    """
    def __init__(self, config):
        self.config = config

        self.anchor_sizes  = np.array(config.anchor_sizes)
        self.anchor_ratios = np.array(config.anchor_ratios)

        self.anchors_count = len(self.anchor_sizes) * len(self.anchor_ratios)


    def build(self, conv_feature_map, im_shape, all_anchors,
               gt_boxes=None, is_training=False):
        pass

    def generate_anchors(self, feature_map_shape):
        """Generate anchor for an image.
        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.
        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.
        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).
        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        """
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]  # width
            grid_height = feature_map_shape[1]  # height
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            # Expand dims to use broadcasting sum.
            all_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1)
            )

            # Flatten
            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors
        """