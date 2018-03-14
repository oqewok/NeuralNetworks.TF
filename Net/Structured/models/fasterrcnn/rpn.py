import tensorflow as tf
import numpy as np

from Structured.models.fasterrcnn.rpn_proposal import RPNProposal
from Structured.utils.operations import *
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
    def __init__(self, config, conv_feats, conv_feats_shape, is_training):
        self.config           = config

        self.img_shape        = self.config.input_shape
        self.conv_feats       = conv_feats
        self.conv_feats_shape = conv_feats_shape

        self.anchor_scales    = np.array(config.anchor_scales)
        self.anchor_ratios    = np.array(config.anchor_ratios)
        self.anchors_count    = len(self.anchor_scales) * len(self.anchor_ratios)

        self.is_training      = is_training

        self.build()

    def build(self):
        img_shape2d             = self.img_shape[:2]
        conv_feats_shape2d      = self.conv_feats_shape[:2]

        # TODO: Переделать генерацию "якорей" в anchors.py.
        # """ Generate the anchors. """
        #self.all_anchors        = generate_anchors(
        #     img_shape2d, conv_feats_shape2d, self.anchor_scales, self.anchor_ratios)

        #self.total_anchor_count = np.count_nonzero(self.anchors, axis=(0, 1, 2)) // 4

        # Get the RPN feature using a simple conv net. Activation function
        # can be set to empty.
        self.rpn_conv_feature       = convolution(
            self.conv_feats, 3, 3, 512, 1, 1, 'rpn_conv', group_id=1)
        self.rpn_feature            = nonlinear(
            self.rpn_conv_feature, 'relu')

        # Then we apply separate convolution layers for classification and regression.

        # rpn_cls_score_original has shape (?, H, W, num_anchors * 2)
        # rpn_bbox_pred_original has shape (?, H, W, num_anchors * 4)
        # where H, W are height and width of the feature map.
        self.rpn_cls_score_original  = convolution(
            self.rpn_feature, 1, 1, 2 * self.anchors_count, 1, 1, 'rpn_cls', group_id=1)
        self.rpn_bbox_pred_original  = convolution(
            self.rpn_feature, 1, 1, 4 * self.anchors_count, 1, 1, 'rpn_regs', group_id=1)

        # Convert (flatten) `rpn_cls_score_original` which has two scalars per
        # anchor per location to be able to apply softmax.
        self.rpn_cls_score = tf.reshape(self.rpn_cls_score_original, [self.config.batch_size, -1, 2])
        #self.rpn_cls_score = tf.reshape(self.rpn_cls_score_original, [-1, 2])

        # Now that `rpn_cls_score` has shape (Batch_size * H * W * num_anchors, 2), we apply
        # softmax to the last dim.
        self.rpn_cls_prob = tf.nn.softmax(self.rpn_cls_score)

        # Flatten bounding box delta prediction for easy manipulation.
        # We end up with `rpn_bbox_pred` having shape (Batch_size * H * W * num_anchors, 4).
        self.rpn_bbox_pred = tf.reshape(self.rpn_bbox_pred_original, [self.config.batch_size, -1, 4])
        #self.rpn_bbox_pred = tf.reshape(self.rpn_bbox_pred_original, [-1, 4])

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redundant ones using Non Maximum Suppression (NMS).
        # TODO: Реализовать класс RPNProposal в rpn_proposal.py
        self.proposal = RPNProposal(
            self.config, self.anchors_count
        )

        # TODO: Реализовать метод, который выдает предсказания координат в rpn_proposal.py
        # self.proposal_prediction = self.proposal.get_obj_proposals(
        #     self.rpn_cls_prob, self.rpn_bbox_pred, all_anchors, img_shape2d)

        pass


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