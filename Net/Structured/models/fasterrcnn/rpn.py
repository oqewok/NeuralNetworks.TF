import tensorflow as tf
import numpy as np

from Structured.models.fasterrcnn.rpn_proposal import RPNProposal
from Structured.models.fasterrcnn.rpn_target import RPNTarget
from Structured.utils.operations import *
from Structured.utils.losses import smooth_l1_loss
from Structured.utils.anchors import generate_anchors

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
    def __init__(self, config, conv_feats, conv_feats_shape, gt_boxes, is_training):
        print("Building RPN.")

        self.config           = config

        self.img_shape        = self.config.input_shape
        self.conv_feats       = conv_feats
        self.conv_feats_shape = conv_feats_shape
        self.reg              = self.config.regularization

        self.gt_boxes         = gt_boxes

        self.anchor_base_size = self.config.anchor_base_size
        self.anchor_scales    = np.array(config.anchor_scales)
        self.anchor_ratios    = np.array(config.anchor_ratios)
        self.anchor_stride    = self.config.anchor_stride
        self.anchors_count    = len(self.anchor_scales) * len(self.anchor_ratios)

        self.l1_sigma         = self.config.l1_sigma
        self.is_training      = is_training

        self.predict()


    def predict(self):
        img_shape2d             = self.img_shape[:2]

        # Get the RPN feature using a simple conv net. Activation function
        # can be set to empty.
        self.rpn_conv_feature       = convolution(
            self.conv_feats, 3, 3, 512, 1, 1, 'rpn_conv', init_w="normal", stddev=0.01, reg=self.reg,  group_id=1
        )
        self.rpn_feature            = nonlinear(
            self.rpn_conv_feature, 'relu'
        )

        # Then we apply separate convolution layers for classification and regression.

        # rpn_cls_score_original has shape (1, H, W, num_anchors * 2)
        # rpn_bbox_pred_original has shape (1, H, W, num_anchors * 4)
        # where H, W are height and width of the feature map.
        self.rpn_cls_score_original  = convolution(
            self.rpn_feature, 1, 1, 2 * self.anchors_count, 1, 1, 'rpn_cls', init_w="normal", stddev=0.01, reg=self.reg, group_id=1
        )
        self.rpn_bbox_pred_original  = convolution(
            self.rpn_feature, 1, 1, 4 * self.anchors_count, 1, 1, 'rpn_regs', init_w="normal", stddev=0.001, reg=self.reg, group_id=1
        )

        # Convert (flatten) `rpn_cls_score_original` which has two scalars per
        # anchor per location to be able to apply softmax.
        self.rpn_cls_score = tf.reshape(
            self.rpn_cls_score_original, [-1, 2]
        )

        # Now that `rpn_cls_score` has shape (H * W * num_anchors, 2), we apply
        # softmax to the last dim.
        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score)

        # Flatten bounding box delta prediction for easy manipulation.
        # We end up with `rpn_bbox_pred` having shape (H * W * num_anchors, 4).
        self.rpn_bbox_pred = tf.reshape(
            self.rpn_bbox_pred_original, [-1, 4]
        )

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redundant ones using Non Maximum Suppression (NMS).
        self.rpn_proposal = RPNProposal(
            self.config, self.anchors_count
        )

        all_anchors = generate_anchors(
            self.anchor_base_size, self.anchor_stride, self.anchor_ratios, self.anchor_scales, tf.shape(self.conv_feats)
        )

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redundant ones using Non Maximum Suppression (NMS).
        self.proposals, self.scores = self.rpn_proposal.get_obj_proposals(
             self.rpn_cls_prob, self.rpn_bbox_pred, all_anchors, img_shape2d
        )

        self.rpn_target = RPNTarget(
            self.config, self.anchors_count
        )

        # Calculate the target values we want to output.
        self.rpn_cls_target, self.rpn_bbox_target, self.rpn_max_overlap = self.rpn_target.get_targets(
            all_anchors, self.gt_boxes, img_shape2d
        )

        # Define loss tensors
        self.rpn_cls_loss, self.rpn_reg_loss = self.loss(
            self.rpn_cls_score, self.rpn_cls_target, self.rpn_bbox_pred, self.rpn_bbox_target
        )

        print("RPN built.")

    def loss(self, rpn_cls_score, rpn_cls_target, rpn_bbox_pred, rpn_bbox_target):
        """
        Returns cost for Region Proposal Network based on:
        Args:
            rpn_cls_score: Score for being an object or not for each anchor
                in the image. Shape: (num_anchors, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                * 1: for positive labels
                * 0: for negative labels
                * -1: for labels we should ignore.
                Shape: (num_anchors, )
            rpn_bbox_target: Bounding box output delta target for rpn.
                Shape: (num_anchors, 4)
            rpn_bbox_pred: Bounding box output delta prediction for rpn.
                Shape: (num_anchors, 4)
        Returns:
            Multiloss between cls probability and bbox target.
        """

        with tf.variable_scope('RPNLoss'):
            # Flatten already flat Tensor for usage as boolean mask filter.
            rpn_cls_target = tf.cast(tf.reshape(
                rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
            # Transform to boolean tensor mask for not ignored.
            labels_not_ignored = tf.not_equal(
                rpn_cls_target, -1, name='labels_not_ignored')

            # Now we only have the labels we are going to compare with the
            # cls probability.
            labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
            cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)

            # We need to transform `labels` to `cls_score` shape.
            # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
            cls_target = tf.one_hot(labels, depth=2)

            # Equivalent to log loss
            cross_entropy_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
                labels=cls_target, logits=cls_score
            )

            # Finally, we need to calculate the regression loss over
            # `rpn_bbox_target` and `rpn_bbox_pred`.
            # We use SmoothL1Loss.
            rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

            # We only care for positive labels (we ignore backgrounds since
            # we don't have any bounding box information for it).
            positive_labels = tf.equal(rpn_cls_target, 1)
            rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
            rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

            # We apply smooth l1 loss as described by the Fast R-CNN paper.
            reg_loss_per_anchor = smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_target, sigma=self.l1_sigma
            )

            rpn_cls_loss = tf.reduce_sum(cross_entropy_per_anchor)
            rpn_reg_loss = tf.reduce_sum(reg_loss_per_anchor)

            # Loss summaries.
            tf.summary.scalar('batch_size', tf.shape(labels)[0], ['rpn'])
            foreground_cls_loss = tf.boolean_mask(
                cross_entropy_per_anchor, tf.equal(labels, 1))
            background_cls_loss = tf.boolean_mask(
                cross_entropy_per_anchor, tf.equal(labels, 0))
            tf.summary.scalar(
                'foreground_cls_loss',
                tf.reduce_mean(foreground_cls_loss), ['rpn'])
            tf.summary.histogram(
                'foreground_cls_loss', foreground_cls_loss, ['rpn'])
            tf.summary.scalar(
                'background_cls_loss',
                tf.reduce_mean(background_cls_loss), ['rpn'])
            tf.summary.histogram(
                'background_cls_loss', background_cls_loss, ['rpn'])
            tf.summary.scalar(
                'foreground_samples', tf.shape(rpn_bbox_target)[0], ['rpn'])

            return rpn_cls_loss, rpn_reg_loss

""" Generate the anchors.
# anchor_scales = [32, 64, 128] => factor = 2.
scales = np.array([32, 64, 128], dtype=int)
ratios = np.array([1, 4, 6], dtype=int)

ih, iw = 600, 800
fh, fw = 39, 51
n      = fh * fw
print("Building the anchors...")

pass
"""