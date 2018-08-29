import numpy as np
import tensorflow as tf

from Structured.utils.operations import *
from Structured.base.base_model import BaseModel
from Structured.nets.mobilenet_v2_1_0_224_feature_extractor import get_mobilenet_v2_1_0_feature_extractor
from Structured.models.ssd.utils import (
    generate_raw_anchors, adjust_bboxes
)
from Structured.utils.bbox_transform import clip_boxes


class SSDModel(BaseModel):
    """SSD: Single Shot MultiBox Detector
    """

    def __init__(self, config):
        super(SSDModel, self).__init__(config)

        self.config = config
        self.num_classes = config.num_classes
        self.anchor_max_scale = config.anchor_max_scale
        self.anchor_min_scale = config.anchor_min_scale
        self.anchor_ratios = np.array(config.anchor_ratios)
        self.input_shape = config.input_shape
        self.reg = config.regularization

        self.anchors_per_point = config.anchors_per_point

        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.config.learning_rate,
            global_step=self.global_step_tensor,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate,
            staircase=True
        )

        self.momentum = self.config.momentum

        self.optimizer_name = self.config.optimizer
        self.optimizer = None

        self.build_model()
        self.init_saver()


    def build_model(self):
        pass
        # Value of training mode
        self.is_training = self.config.is_training

        # GT boxes tensor.
        self.gt_boxes = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="gt_boxes")

        # Tensor for training mode description. If true => training mode, else => evaluation mode.
        self.is_training_tensor = tf.placeholder(tf.bool, name="is_train")

        if self.config.basic_cnn == "mobilenet_v2_1.0_224":
            self.inputs_tensor, self.feature_maps = get_mobilenet_v2_1_0_feature_extractor()

        # Build a MultiBox predictor on top of each feature layer and collect
        # the bounding box offsets and the category score logits they produce
        bbox_offsets_list = []
        class_scores_list = []

        for i, feat_map in enumerate(self.feature_maps):
            multibox_predictor_name = 'MultiBox_{}'.format(i)
            with tf.name_scope(multibox_predictor_name):
                num_anchors = self.anchors_per_point[i]

                # Predict bbox offsets
                bbox_offsets_layer = convolution(
                    feat_map, 3, 3, num_anchors * 4, 1, 1, 'rpn_conv', init_w="normal", stddev=0.001, reg=self.reg,  group_id=1
                )
                bbox_offsets_flattened = tf.reshape(
                    bbox_offsets_layer, [-1, 4]
                )
                bbox_offsets_list.append(bbox_offsets_flattened)

                # Predict class scores
                class_scores_layer = convolution(
                    feat_map, 3, 3, num_anchors * (self.num_classes + 1), 1, 1, 'rpn_conv', init_w="normal",
                    stddev=0.001, reg=self.reg, group_id=1
                )
                class_scores_flattened = tf.reshape(
                    class_scores_layer, [-1, self.num_classes + 1]
                )
                class_scores_list.append(class_scores_flattened)

            bbox_offsets = tf.concat(
                bbox_offsets_list, axis=0, name='concatenate_all_bbox_offsets'
            )
            class_scores = tf.concat(
                class_scores_list, axis=0, name='concatenate_all_class_scores'
            )
            class_probabilities = tf.nn.softmax(
                class_scores, axis=-1, name='class_probabilities_softmax'
            )

            # Generate anchors (generated only once, therefore we use numpy)
            raw_anchors_per_featmap = generate_raw_anchors(
                self.feature_maps, self.anchor_min_scale, self.anchor_max_scale,
                self.anchor_ratios, self.anchors_per_point
            )
            anchors_list = []

            for i, feat_map in enumerate(self.feature_maps):
                feat_map_shape = feat_map.shape.as_list()[1:3]
                scaled_bboxes = adjust_bboxes(
                    raw_anchors_per_featmap[i], feat_map_shape[0],
                    feat_map_shape[1], self.input_shape[0], self.input_shape[1]
                )

                clipped_bboxes = clip_boxes(scaled_bboxes, self.input_shape[0:2])
                anchors_list.append(clipped_bboxes)
            anchors = np.concatenate(anchors_list, axis=0)
            anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

            # This is the dict we'll return after filling it with SSD's results
            prediction_dict = {}

            """
            # Generate targets for training
            if self.gt_boxes is not None:
                gt_boxes = tf.cast(self.gt_boxes, tf.float32)

                # Generate targets
                target_creator = SSDTarget(
                    self.num_classes, self.config.target, self.config.variances
                )
                class_targets, bbox_offsets_targets = target_creator(
                    class_probabilities, anchors, gt_boxes
                )
            """