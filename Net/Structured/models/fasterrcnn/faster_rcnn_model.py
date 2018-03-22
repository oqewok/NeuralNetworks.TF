from Structured.base.base_model import BaseModel
from Structured.nets.vgg16 import build_basic_vgg16
from Structured.models.fasterrcnn.rpn import RPN
from Structured.models.fasterrcnn.roi_pooling import ROIPooling

import tensorflow as tf


class FasterRCNNModel(BaseModel):
    def __init__(self, config):
        super(FasterRCNNModel, self).__init__(config)

        self.learning_rate  = tf.train.exponential_decay(
            learning_rate=self.config.learning_rate,
            global_step=self.global_step_tensor,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate,
            staircase=True
        )

        self.optimizer_name = self.config.optimizer
        self.optimizer      = None

        self.build_model()
        self.init_saver()


    def build_model(self):
        # here you predict the tensorflow graph of any model you want and also define the loss.

        print("Building model.")

        # Value of training mode
        self.is_training = self.config.is_training

        # Inputs or X. Tensor for the batch of images.
        self.inputs_tensor      = None

        # GT boxes tensor.
        self.gt_boxes = tf.placeholder(shape=[None, 4], dtype=tf.int32, name="gt_boxes")

        # Tensor for training mode description. If true => training mode, else => evaluation mode.
        self.is_training_tensor = None

        # convolution features after basic CNN feature extraction
        self.conv_feats_tensor  = None

        # convolution features shape (ndarray).
        self.conv_feats_shape   = None

        # Build basic CNN for feature extraction. Default is VGG16.
        ''' For example:
        elif self.config.basic_cnn == "resnet50":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_resnet50(
            self.config)'''
        if self.config.basic_cnn == "vgg16":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)
        else:
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)

        # Build RPN
        self.rpn = RPN(
            self.config, self.conv_feats_tensor, self.conv_feats_shape, gt_boxes=self.gt_boxes, is_training=self.is_training_tensor
        )

        # Predicted RPN objectness scores
        self.rpn_cls_scores     = self.rpn.scores
        # Predicted RPN objects' bounding boxes
        self.rpn_roi_proposals  = self.rpn.proposals

        # RPN loss
        self.rpn_loss = self.rpn.loss

        self.roi_pool = ROIPooling(
            self.config, self.rpn_roi_proposals, self.conv_feats_tensor, self.config.input_shape
        )

        self.roi_pooled_proposals = self.roi_pool.roi_pooled_features

        if self.optimizer_name == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss=self.rpn_loss, global_step=self.global_step_tensor
            )

        print("Model built.")
        pass


    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


"""
from Structured.utils.config import process_config
config = process_config("E:/Study/Mallenom/NeuralNetworks.TF/Net/Structured/configs/fastercnn.json")
model = FasterRCNNModel(config)
pass
"""