import tensorflow as tf
import numpy as np

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

        """ Generate the anchors. """
        self.anchors            = generate_anchors(
            img_shape2d, conv_feats_shape2d, self.anchor_scales, self.anchor_ratios)

        self.total_anchor_count = np.count_nonzero(self.anchors, axis=(0, 1, 2)) // 4

        self.bn                 = self.config.use_batch_norm

        self.feats              = tf.placeholder(tf.float32, [None] + self.conv_feats_shape)
        self.gt_anchor_labels   = tf.placeholder(tf.int32,   [None, self.total_anchor_count])
        self.gt_anchor_regs     = tf.placeholder(tf.float32, [None, self.total_anchor_count, 4])
        self.anchor_masks       = tf.placeholder(tf.float32, [None, self.total_anchor_count])
        self.anchor_weights     = tf.placeholder(tf.float32, [None, self.total_anchor_count])
        self.anchor_reg_masks   = tf.placeholder(tf.float32, [None, self.total_anchor_count])

        # Compute the RoI proposals
        all_rpn_logits          = []
        all_rpn_regs            = []

        current_feats           = self.feats

        if self.config.basic_cnn == 'vgg16':
            kernel_sizes = [10, 10]
        else:
            kernel_sizes = [5, 5]

        for i in range(2):
            label_i = '_' + str(i)
            rpn1 = convolution(current_feats, kernel_sizes[0], kernel_sizes[1], 512, 1, 1, 'rpn1' + label_i, group_id=1)
            rpn1 = nonlinear(rpn1, 'relu')
            rpn1 = dropout(rpn1, 0.5, self.is_training)

            for j in range(9):
                label_ij = str(i) + '_' + str(j)

                rpn_logits = convolution(rpn1, 1, 1, 2, 1, 1, 'rpn_logits' + label_ij, group_id=1)
                rpn_logits = tf.reshape(rpn_logits, [self.config.batch_size, -1, 2])
                all_rpn_logits.append(rpn_logits)

                rpn_regs = convolution(rpn1, 1, 1, 4, 1, 1, 'rpn_regs' + label_ij, group_id=1)
                rpn_regs = tf.clip_by_value(rpn_regs, -0.2, 0.2)
                rpn_regs = tf.reshape(rpn_regs, [self.config.batch_size, -1, 4])
                all_rpn_regs.append(rpn_regs)

            if i < 1:
                current_feats = max_pool(current_feats, 2, 2, 2, 2, 'rpn_pool' + label_i)

        all_rpn_logits          = tf.concat(1, all_rpn_logits)
        all_rpn_regs            = tf.concat(1, all_rpn_regs)

        all_rpn_logits          = tf.reshape(all_rpn_logits, [-1, 2])
        all_rpn_regs            = tf.reshape(all_rpn_regs, [-1, 4])

        # Compute the loss function
        self.gt_anchor_labels   = tf.reshape(self.gt_anchor_labels, [-1])
        self.gt_anchor_regs     = tf.reshape(self.gt_anchor_regs, [-1, 4])
        self.anchor_masks       = tf.reshape(self.anchor_masks, [-1])
        self.anchor_weights     = tf.reshape(self.anchor_weights, [-1])
        self.anchor_reg_masks   = tf.reshape(self.anchor_reg_masks, [-1])

        loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(all_rpn_logits, self.gt_anchor_labels) * self.anchor_masks
        loss0 = tf.reduce_sum(loss0 * self.anchor_weights) / tf.reduce_sum(self.anchor_weights)

        w = self.l2_loss(all_rpn_regs, self.gt_anchor_regs) * self.anchor_reg_masks
        z = tf.reduce_sum(self.anchor_reg_masks)
        loss0 = tf.cond(tf.less(0.0, z), lambda: loss0 + self.config.rpn_reg_weight * tf.reduce_sum(w) / z, lambda: loss0)

        loss1 = self.config.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1

        # Build the optimizer
        if self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.config.learning_rate, self.config.momentum)
        elif self.config.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate, self.config.decay, self.config.momentum)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        #opt_op = optimizer.minimize(loss, global_step=self.global_step)
        opt_op          = optimizer.minimize(loss)

        rpn_probs       = tf.nn.softmax(all_rpn_logits)
        rpn_scores      = tf.squeeze(tf.slice(rpn_probs, [0, 1], [-1, 1]))
        rpn_scores      = tf.reshape(rpn_scores, [self.config.batch_size, self.total_anchor_count])
        rpn_regs        = tf.reshape(all_rpn_regs, [self.config.batch_size, self.total_anchor_count, 4])

        self.rpn_loss   = loss
        self.rpn_loss0  = loss0
        self.rpn_loss1  = loss1
        self.rpn_opt_op = opt_op

        self.rpn_scores = rpn_scores
        self.rpn_regs   = rpn_regs


    def l2_loss(self, s, t):
        """ L2 loss function. """
        d = s - t
        x = d * d
        loss = tf.reduce_sum(x, 1)
        return loss
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