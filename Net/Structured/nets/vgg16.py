import tensorflow as tf

from Structured.utils.config import process_config
from Structured.utils.operations import *


def build_basic_vgg16(config):
    """ Build the basic VGG16 net. """
    print("Building the basic VGG16 net...")
    bn = config.use_batch_norm

    H, W, C = config.input_shape

    imgs = tf.placeholder(tf.float32, [None, H, W, C])
    is_train = tf.placeholder(tf.bool)

    conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
    conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
    conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
    conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
    pool1_feats   = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

    conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
    conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
    conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
    conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
    pool2_feats   = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

    conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
    conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
    conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
    conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
    conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
    conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
    pool3_feats   = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

    conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
    conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
    conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
    conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
    conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
    conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
    pool4_feats   = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

    conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
    conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
    conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
    conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
    conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
    conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

    conv_feats       = conv5_3_feats
    conv_feats_shape = np.stack((
        int(conv_feats.shape[1]), int(conv_feats.shape[2]), int(conv_feats.shape[3])))

    print("Basic VGG16 net built.")

    return imgs, is_train, conv_feats, conv_feats_shape

'''
config = process_config("E:/Study/Mallenom/NeuralNetworks.TF/Net/Structured/configs/fastercnn.json")
build_basic_vgg16(config)
'''