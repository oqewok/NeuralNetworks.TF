from Structured.base.base_model import BaseModel
from Structured.nets.vgg16 import build_basic_vgg16
from Structured.models.vgg_pretrained import get_vgg16_pretrained
from Structured.models.fasterrcnn.rpn import RPN
from Structured.models.fasterrcnn.rcnn import RCNN

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
        #self.learning_rate = self.config.learning_rate

        self.momentum       = self.config.momentum

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
        self.gt_boxes = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="gt_boxes")

        # Tensor for training mode description. If true => training mode, else => evaluation mode.
        self.is_training_tensor = tf.placeholder(tf.bool, name="is_train")

        # convolution features after basic CNN feature extraction
        self.conv_feats_tensor  = None

        # convolution features shape (ndarray).
        self.conv_feats_shape   = None

        self.with_rcnn = self.config.with_rcnn
        # Build basic CNN for feature extraction. Default is VGG16.
        ''' For example:
        elif self.config.basic_cnn == "resnet50":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_resnet50(
            self.config)'''
        if self.config.basic_cnn == "vgg16":
            #self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
            #    self.config)
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_vgg16_pretrained()

        # Build RPN
        self.rpn = RPN(
            self.config, self.conv_feats_tensor, self.conv_feats_shape, gt_boxes=self.gt_boxes, is_training=self.is_training_tensor
        )

        # Predicted RPN objectness scores
        self.rpn_cls_scores     = self.rpn.scores
        # Predicted RPN objects' bounding boxes
        self.rpn_roi_proposals  = self.rpn.proposals


        # RCNN
        #proposals = tf.stop_gradient(self.rpn_roi_proposals)
        if self.with_rcnn:
            self.rcnn = RCNN(
                self.config, self.conv_feats_tensor, self.rpn_roi_proposals,
                self.config.input_shape, gt_boxes=self.gt_boxes, is_training=self.is_training_tensor
            )

            # Predicted RCNN
            self.objects                = self.rcnn.objects
            self.proposal_label         = self.rcnn.proposal_label
            self.proposal_label_prob    = self.rcnn.proposal_label_prob


        with tf.name_scope('losses'):
            # RPN losses
            self.rpn_cls_loss, self.rpn_reg_loss = self.rpn.rpn_cls_loss, self.rpn.rpn_reg_loss
            if self.with_rcnn:
                #RCNN losses
                self.rcnn_cls_loss, self.rcnn_reg_loss = self.rcnn.rcnn_cls_loss, self.rcnn.rcnn_reg_loss
                #self.loss = self.rcnn_reg_loss

                all_losses_dict = {
                        "rpn_cls_loss"  : self.rpn_cls_loss,
                        "rpn_reg_loss"  : self.rpn_reg_loss,
                        "rcnn_cls_loss" : self.rcnn_cls_loss,
                        "rcnn_reg_loss" : self.rcnn_reg_loss,
                }
            else:
                all_losses_dict = {
                    "rpn_cls_loss": self.rpn_cls_loss,
                    "rpn_reg_loss": self.rpn_reg_loss,
                }

            for loss_name, loss_tensor in all_losses_dict.items():
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=['fastercnn_losses']
                )
                # We add losses to the losses collection instead of manually
                # summing them just in case somebody wants to use it in another
                # place.
                tf.losses.add_loss(loss_tensor)

                # Regularization loss is automatically saved by TensorFlow, we log
                # it differently so we can visualize it independently.

            self.loss = tf.losses.get_total_loss()
            self.regularization_loss = tf.losses.get_regularization_loss()

        optimizer = None
        if self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )
        elif self.optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            )

        self.optimizer = optimizer.minimize(
            loss=self.loss, global_step=self.global_step_tensor
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