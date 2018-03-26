import tensorflow as tf

from Structured.models.fasterrcnn.roi_pooling import ROIPooling
from Structured.models.fasterrcnn.rcnn_proposal import RCNNProposal
from Structured.models.fasterrcnn.rcnn_target import RCNNTarget
from Structured.utils.losses import smooth_l1_loss
from Structured.utils.operations import *


class RCNN:
    """RCNN: Region-based Convolutional Neural Network.
        Given region proposals (bounding boxes on an image) and a feature map of
        that image, RCNN adjusts the bounding boxes and classifies each region as
        either background or a specific object class.
        Steps:
            1. Region of Interest Pooling. Extract features from the feature map
               (based on the proposals) and convert into fixed size tensors
               (applying extrapolation).
            2. Two fully connected layers generate a smaller tensor for each
               region.
            3. A fully conected layer outputs the probability distribution over the
               classes (plus a background class), and another fully connected layer
               outputs the bounding box regressions (one 4-d regression for each of
               the possible classes).
        Using the class probability, filter regions classified as background. For
        the remaining regions, use the class probability together with the
        corresponding bounding box regression offsets to generate the final object
        bounding boxes, with classes and probabilities assigned.
        """
    pass

    def __init__(self, config, conv_feature_map, proposals, img_shape, gt_boxes=None, is_training=False):
        self.config = config
        self.num_classes    = self.config.num_classes

        self.reg            = self.config.regularization
        self.l1_sigma       = self.config.rcnn_l1_sigma
        self.use_mean       = self.config.use_mean
        self.dropout_prob   = self.config.dropout

        self.build(conv_feature_map, proposals, img_shape, gt_boxes, is_training)

    def build(self, conv_feature_map, proposals, img_shape, gt_boxes, is_training):
        """
                Classifies & refines proposals based on the pooled feature map.
                Args:
                    conv_feature_map: The feature map of the image, extracted
                        using the pretrained network.
                        Shape: (num_proposals, pool_height, pool_width, 512).
                    proposals: A Tensor with the bounding boxes proposed by the RPN.
                        Shape: (total_num_proposals, 4).
                        Encoding: (x1, y1, x2, y2).
                    img_shape: A Tensor with the shape of the image in the form of
                        (image_height, image_width).
                    gt_boxes (optional): A Tensor with the ground truth boxes of the
                        image.
                        Shape: (total_num_gt, 5).
                        Encoding: (x1, y1, x2, y2, label).
                    is_training (optional): A boolean to determine if we are just using
                        the module for training or just inference.
                Returns:
                    prediction_dict: a dict with the object predictions.
                        It should have the keys:
                        objects:
                        labels:
                        probs:
                        rcnn:
                        target:
                """
        print("Building RCNN")

        self.roi_pool = ROIPooling(
            self.config, proposals, conv_feature_map, img_shape
        )

        self.roi_pooled_features = self.roi_pool.roi_pooled_features

        if self.use_mean:
            # We avg our height and width dimensions for a more
            # "memory-friendly" Tensor.
            self.roi_pooled_features = tf.reduce_mean(self.roi_pooled_features, axis=(1, 2))
        # else:
        #     s = self.roi_pooled_features.shape
        #     self.roi_pooled_features = tf.reshape(self.roi_pooled_features, [-1, s[1]*s[2]*s[3]])

        self.build_fc_layers(self.roi_pooled_features, is_training)

        # Get final objects proposals based on the probabilty, the offsets and
        # the original proposals.
        self.rcnn_proposal = RCNNProposal(self.config)

        # objects, objects_labels, and objects_labels_prob are the only keys
        # that matter for drawing objects.
        self.objects, self.proposal_label, self.proposal_label_prob = self.rcnn_proposal.get_proposals(
            proposals, self.bbox_offsets, self.cls_prob, img_shape
        )

        self.rcnn_target = RCNNTarget(self.config)

        self.cls_target, self.bbox_offsets_target = self.rcnn_target.get_targets(
            proposals, gt_boxes
        )

        # Define loss tensors
        self.rcnn_cls_loss, self.rcnn_reg_loss = self.loss(
            self.cls_scores, self.cls_target, self.bbox_offsets, self.bbox_offsets_target
        )
        print("RCNN built.")

    def build_fc_layers(self, proposals, is_training):
        # We define layers as an array since they are simple fully connected
        # ones and it should be easy to tune it from the network config.

        # Compute the RoI classification results
        fc6_feats = fully_connected(proposals, 4096, 'rcn_fc6', init_w='variance_scaling', reg=self.reg, group_id=2)
        fc6_feats = nonlinear(fc6_feats, 'relu')
        fc6_feats = dropout(fc6_feats, self.dropout_prob, is_training)

        fc7_feats = fully_connected(fc6_feats, 4096, 'rcn_fc7', init_w='variance_scaling', reg=self.reg, group_id=2)
        fc7_feats = nonlinear(fc7_feats, 'relu')
        fc7_feats = dropout(fc7_feats, self.dropout_prob, is_training)

        # We define the classifier layer having a num_classes + 1 background
        # since we want to be able to predict if the proposal is background as
        # well.
        self.cls_scores = fully_connected(fc7_feats, self.num_classes, 'fc_cls', init_w='normal', reg=self.reg, group_id=2)
        #self.cls_scores = fully_connected(fc7_feats, self.num_classes + 1, 'fc_cls', init_w='normal', group_id=2)
        self.cls_prob = tf.nn.softmax(self.cls_scores)

        # The bounding box adjustment layer has 4 times the number of classes
        # We choose which to use depending on the output of the classifier
        # layer
        self.bbox_offsets = fully_connected(fc7_feats, self.num_classes * 4, 'fc_bbox', init_w='normal', reg=self.reg, group_id=2)


    def loss(self, cls_score, cls_target, bbox_offsets, bbox_offsets_target):
        """
                Returns cost for RCNN based on:
                Args:
                    prediction_dict with keys:
                        rcnn:
                            cls_score: shape (num_proposals, num_classes + 1)
                                Has the class scoring for each the proposals. Classes
                                are 1-indexed with 0 being the background.
                            bbox_offsets: shape (num_proposals, num_classes * 4)
                                Has the offset for each proposal for each class.
                                We have to compare only the proposals labeled with the
                                offsets for that label.
                        target:
                            cls_target: shape (num_proposals,)
                                Has the correct label for each of the proposals.
                                0 => background
                                1..n => 1-indexed classes
                            bbox_offsets_target: shape (num_proposals, 4)
                                Has the true offset of each proposal for the true
                                label.
                                In case of not having a true label (non-background)
                                then it's just zeroes.
                Returns:
                    loss_dict with keys:
                        rcnn_cls_loss: The cross-entropy or log-loss of the
                            classification tasks between then num_classes + background.
                        rcnn_reg_loss: The smooth L1 loss for the bounding box
                            regression task to adjust correctly labeled boxes.
                """

        with tf.name_scope('RCNNLoss'):
            # Cast target explicitly as int32.
            cls_target = tf.cast(
                cls_target, tf.int32
            )

            # First we need to calculate the log loss between cls_prob and
            # cls_target

            # We only care for the targets that are >= 0
            not_ignored = tf.reshape(tf.greater_equal(
                cls_target, 0), [-1], name='not_ignored')
            # We apply boolean mask to score, prob and target.
            cls_score_labeled = tf.boolean_mask(
                cls_score, not_ignored, name='cls_score_labeled')
            # cls_prob_labeled = tf.boolean_mask(
            #     cls_prob, not_ignored, name='cls_prob_labeled')
            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')

            tf.summary.scalar(
                'batch_size',
                tf.shape(cls_score_labeled)[0], ['rcnn']
            )

            # Transform to one-hot vector
            cls_target_one_hot = tf.one_hot(
                #cls_target_labeled, depth=self.num_classes + 1,
                cls_target_labeled, depth=self.num_classes,
                name='cls_target_one_hot'
            )

            # We get cross entropy loss of each proposal.
            cross_entropy_per_proposal = (
                tf.nn.softmax_cross_entropy_with_logits(
                    #labels=tf.stop_gradient(cls_target_one_hot),
                    labels=cls_target_one_hot,
                    logits=cls_score_labeled
                )
            )

            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_target`.

            # We only want the non-background labels bounding boxes.
            not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
            bbox_offsets_labeled = tf.boolean_mask(
                bbox_offsets, not_ignored, name='bbox_offsets_labeled')
            bbox_offsets_target_labeled = tf.boolean_mask(
                bbox_offsets_target, not_ignored,
                name='bbox_offsets_target_labeled'
            )

            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')
            # `cls_target_labeled` is based on `cls_target` which has
            # `num_classes` + 1 classes.
            # for making `one_hot` with depth `num_classes` to work we need
            # to lower them to make them 0-index.

            cls_target_labeled = cls_target_labeled - 1

            cls_target_one_hot = tf.one_hot(
                cls_target_labeled, depth=self.num_classes,
                name='cls_target_one_hot'
            )

            # cls_target now is (num_labeled, num_classes)
            bbox_flatten = tf.reshape(
                bbox_offsets_labeled, [-1, 4], name='bbox_flatten')

            # We use the flatten cls_target_one_hot as boolean mask for the
            # bboxes.
            cls_flatten = tf.cast(tf.reshape(
                cls_target_one_hot, [-1]), tf.bool, 'cls_flatten_as_bool')

            bbox_offset_cleaned = tf.boolean_mask(
                bbox_flatten, cls_flatten, 'bbox_offset_cleaned')

            # Calculate the smooth l1 loss between the "cleaned" bboxes
            # offsets (that means, the useful results) and the labeled
            # targets.
            reg_loss_per_proposal = smooth_l1_loss(
                bbox_offset_cleaned, bbox_offsets_target_labeled,
                sigma=self.l1_sigma
            )

            tf.summary.scalar(
                'rcnn_foreground_samples',
                tf.shape(bbox_offset_cleaned)[0], ['rcnn']
            )

            rcnn_cls_loss = tf.reduce_mean(cross_entropy_per_proposal)
            rcnn_reg_loss = tf.reduce_mean(reg_loss_per_proposal)

            return rcnn_cls_loss, rcnn_reg_loss
