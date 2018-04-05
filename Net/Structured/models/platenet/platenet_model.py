from Structured.base.base_model import BaseModel
from Structured.nets.vgg16_full import build_basic_vgg16
from Structured.utils.losses import *

import tensorflow as tf
import numpy as np

class PlateNetModel(BaseModel):
    def __init__(self, config):
        super(PlateNetModel, self).__init__(config)

        self.learning_rate  = tf.train.exponential_decay(
            learning_rate=self.config.learning_rate,
            global_step=self.global_step_tensor,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate,
            staircase=True
        )

        # self.learning_rate = self.config.learning_rate

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

        # Tensor for training mode description. If true => training mode, else => evaluation mode.
        self.is_train = None

        # Build basic CNN for feature extraction. Default is VGG16.
        ''' For example:
        elif self.config.basic_cnn == "resnet50":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_resnet50(
            self.config)'''

        if self.config.basic_cnn == "vgg16":
            self.inputs_tensor, self.cls_prob, self.labels, self.gt_boxes, self.is_train = build_basic_vgg16(
                self.config)

        self.labels_one_hot = tf.one_hot(self.labels, depth=self.config.num_classes)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels_one_hot, logits=self.cls_prob, name="loss"
            )
        )

        #self.bbox_loss = mse(self.bboxes, self.bbox_reg)

        # loss = tf.reduce_mean(
        #     tf.losses.mean_squared_error(self.gt_boxes_norm, self.bbox_reg_norm)
        # )

        #tf.losses.add_loss(loss)
        #self.loss = tf.losses.get_total_loss()

        #self.loss = self.cls_loss + self.bbox_loss

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
            loss=self.loss , global_step=self.global_step_tensor
        )

        self.correct_prediction = tf.equal(tf.argmax(self.cls_prob, 1), tf.argmax(self.labels_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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