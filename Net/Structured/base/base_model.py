import tensorflow as tf
import os

from Structured.nets.vgg16_pretrained import get_vgg16_pretrained
from Structured.nets.mobilenet_v1_1_0_224_pretrained import get_mobilenet_v1_1_0_pretrained
from Structured.nets.mobilenet_v1_0_75_224_pretrained import get_mobilenet_v1_0_75_pretrained
from Structured.nets.mobilenet_v1_0_5_224_pretrained import get_mobilenet_v1_0_5_pretrained
from Structured.nets.mobilenet_v1_0_25_224_pretrained import get_mobilenet_v1_0_25_pretrained
from Structured.nets.mobilenet_v2_0_75_224_pretrained import get_mobilenet_v2_0_75_pretrained
from Structured.nets.mobilenet_v2_1_0_224_pretrained import get_mobilenet_v2_1_0_pretrained
from Structured.nets.mobilenet_v2_1_4_224_pretrained import get_mobilenet_v2_1_4_pretrained
from Structured.nets.resnet50_v2_pretrained import get_resnet_v2_pretrained

class BaseModel:
    def __init__(self, config):
        self.config = config

        if self.config.basic_cnn == "vgg16":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_vgg16_pretrained()
        elif self.config.basic_cnn == "mobilenet_v1_1.0_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v1_1_0_pretrained()
        elif self.config.basic_cnn == "mobilenet_v1_0.75_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v1_0_75_pretrained()
        elif self.config.basic_cnn == "mobilenet_v1_0.5_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v1_0_5_pretrained()
        elif self.config.basic_cnn == "mobilenet_v1_0.25_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v1_0_25_pretrained()
        elif self.config.basic_cnn == "mobilenet_v2_0.75_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v2_0_75_pretrained()
        elif self.config.basic_cnn == "mobilenet_v2_1.0_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v2_1_0_pretrained()
        elif self.config.basic_cnn == "mobilenet_v2_1.4_224":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_mobilenet_v2_1_4_pretrained()
        elif self.config.basic_cnn == "resnet50_v2":
            self.inputs_tensor, self.conv_feats_tensor, self.conv_feats_shape = get_resnet_v2_pretrained()

        # init the global step
        self.init_global_step()

        self.cur_epoch_tensor = None

        # init the epoch counter
        self.init_cur_epoch()

        self.saver            = None


    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step_tensor)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.config.checkpoint_dir, self.config.exp_name))
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step2'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step2')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
