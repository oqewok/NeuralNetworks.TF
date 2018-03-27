import tensorflow as tf

from Structured.utils.operations import *
from Structured.utils.img_preproc import preprocess


def build_basic_vgg16(config):
    """ Build the basic VGG16 net. """
    print("Building the basic VGG16 net...")
    bn = config.use_batch_norm
    dropout_prob = config.dropout
    reg = config.regularization
    num_classes = config.num_classes
    H, W, C = config.input_shape

    inputs = tf.placeholder(tf.float32, [None, H, W, C], name="inputs")
    inputs_norm = tf.divide(inputs, 255)
    is_train = tf.placeholder(tf.bool, name="is_train")

    conv1_1_feats = convolution(inputs_norm, 3, 3, 64, 1, 1, 'conv1_1', init_w="normal", stddev=0.01, reg=reg)
    conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
    conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2', init_w="normal", stddev=0.01, reg=reg)
    conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
    pool1_feats   = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

    conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1', init_w="normal", stddev=0.01, reg=reg)
    conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
    conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2', init_w="normal", stddev=0.01, reg=reg)
    conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
    pool2_feats   = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

    conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1', init_w="normal", stddev=0.01, reg=reg)
    conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
    conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2', init_w="normal", stddev=0.01, reg=reg)
    conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
    conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3', init_w="normal", stddev=0.01, reg=reg)
    conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
    pool3_feats   = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

    conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1', init_w="normal", stddev=0.01, reg=reg)
    conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
    conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2', init_w="normal", stddev=0.01, reg=reg)
    conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
    conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3', init_w="normal", stddev=0.01, reg=reg)
    conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
    pool4_feats   = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

    conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1', init_w="normal", stddev=0.01, reg=reg)
    conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
    conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2', init_w="normal", stddev=0.01, reg=reg)
    conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
    conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3', init_w="normal", stddev=0.01, reg=reg)
    conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

    conv5_3_feats_flat = tf.reshape(conv5_3_feats, shape=[-1, 4 * 8 * 512])

    fc6_feats = fully_connected(conv5_3_feats_flat, 4096, 'rcn_fc6', init_w='variance_scaling', reg=reg)
    fc6_feats = nonlinear(fc6_feats, 'relu')
    fc6_feats = dropout(fc6_feats, dropout_prob, is_train)

    fc7_feats = fully_connected(fc6_feats, 4096, 'rcn_fc7', init_w='variance_scaling', reg=reg)
    fc7_feats = nonlinear(fc7_feats, 'relu')
    fc7_feats = dropout(fc7_feats, dropout_prob, is_train)

    # We define the classifier layer having a num_classes + 1 background
    # since we want to be able to predict if the proposal is background as
    # well.
    cls_scores = fully_connected(fc7_feats, num_classes, 'fc_cls', init_w='normal', reg=reg)
    # self.cls_scores = fully_connected(fc7_feats, self.num_classes + 1, 'fc_cls', init_w='normal', group_id=2)
    cls_prob = tf.nn.softmax(cls_scores)

    print("Basic VGG16 net built.")

    return inputs, is_train, cls_prob


if __name__ == '__main__':
    from Structured.utils.config import process_config

    config = process_config("C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\configs\\platenet.json")
    build_basic_vgg16(config)
