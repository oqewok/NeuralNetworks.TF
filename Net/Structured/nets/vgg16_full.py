import tensorflow as tf

from Structured.utils.operations import *
from Structured.utils.img_preproc import preprocess

def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_basic_vgg16(config):
    """ Build the basic VGG16 net. """
    print("Building the basic VGG16 net...")
    bn = config.use_batch_norm
    dropout_prob = config.dropout
    reg = config.regularization
    num_classes = config.num_classes
    H, W, C = config.input_shape

    inputs = tf.placeholder(tf.float32, [None, H, W, C], name="inputs")
    #inputs_norm = tf.divide(inputs, 255)
    #is_train = tf.placeholder(tf.bool, name="is_train")

    gt_boxes = tf.placeholder(tf.float32, [None, 4], name="gt_boxes")
    #gt_boxes_norm = gt_boxes / (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H) - 1.

    # Convolution Layer 1
    W_conv1 = weight_variable("w1", [3, 3, 1, 32])
    b_conv1 = bias_variable("b1", [32])
    h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Convolution Layer 2
    W_conv2 = weight_variable("w2", [2, 2, 32, 64])
    b_conv2 = bias_variable("b2", [64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Convolution Layer 3
    W_conv3 = weight_variable("w3", [2, 2, 64, 128])
    b_conv3 = bias_variable("b3", [128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # Dense layer 1
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 16 * 128])
    W_fc1 = weight_variable("w4", [8 * 16 * 128, 500])
    b_fc1 = bias_variable("b4", [500])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    # Dense layer 2
    W_fc2 = weight_variable("w5", [500, 500])
    b_fc2 = bias_variable("b5", [500])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # Output layer
    W_out = weight_variable("w6", [500, 4])
    b_out = bias_variable("b6", [4])

    bbox_reg_norm = tf.matmul(h_fc2, W_out) + b_out

    # conv1_1_feats = convolution(inputs_norm, 3, 3, 64, 1, 1, 'conv1_1', init_w="xavier", stddev=0.01, reg=reg)
    # conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
    # conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2', init_w="xavier", stddev=0.01, reg=reg)
    # conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
    # pool1_feats   = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')
    #
    # conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1', init_w="xavier", stddev=0.01, reg=reg)
    # conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
    # conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2', init_w="xavier", stddev=0.01, reg=reg)
    # conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
    # pool2_feats   = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')
    #
    # conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1', init_w="xavier", stddev=0.01, reg=reg)
    # conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
    # conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2', init_w="xavier", stddev=0.01, reg=reg)
    # conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
    # conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3', init_w="xavier", stddev=0.01, reg=reg)
    # conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
    # pool3_feats   = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')
    #
    # conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1', init_w="xavier", stddev=0.01, reg=reg)
    # conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
    # conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2', init_w="xavier", stddev=0.01, reg=reg)
    # conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
    # conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3', init_w="xavier", stddev=0.01, reg=reg)
    # conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
    # pool4_feats   = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')
    #
    # conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1', init_w="xavier", stddev=0.01, reg=reg)
    # conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
    # conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2', init_w="xavier", stddev=0.01, reg=reg)
    # conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
    # conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3', init_w="xavier", stddev=0.01, reg=reg)
    # conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')
    #
    # conv5_3_feats_flat = tf.reshape(conv5_3_feats, shape=[-1, 4 * 8 * 512])
    #
    # fc6_feats = fully_connected(conv5_3_feats_flat, 500, 'rcn_fc6', init_w='xavier', reg=reg)
    # fc6_feats = nonlinear(fc6_feats, 'relu')
    # fc6_feats = dropout(fc6_feats, dropout_prob, is_train)
    #
    # fc7_feats = fully_connected(fc6_feats, 500, 'rcn_fc7', init_w='xavier', reg=reg)
    # fc7_feats = nonlinear(fc7_feats, 'relu')
    # fc7_feats = dropout(fc7_feats, dropout_prob, is_train)
    #
    # bbox_reg_norm  = fully_connected(fc7_feats, 4, 'fc_cls', init_w='xavier', reg=reg)

    #bbox_reg = (bbox_reg_norm + 1.0) * (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H)
    """
    # We define the classifier layer having a num_classes + 1 background
    # since we want to be able to predict if the proposal is background as
    # well.
    cls_scores = fully_connected(fc7_feats, num_classes, 'fc_cls', init_w='xavier', reg=reg)
    # self.cls_scores = fully_connected(fc7_feats, self.num_classes + 1, 'fc_cls', init_w='normal', group_id=2)
    cls_prob = tf.nn.softmax(cls_scores)
    """
    print("Basic VGG16 net built.")

    # return inputs, is_train, bbox_reg, bbox_reg_norm, gt_boxes , gt_boxes_norm, inputs_norm
    return inputs, bbox_reg_norm, gt_boxes


if __name__ == '__main__':
    from Structured.utils.config import process_config

    config = process_config("C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\configs\\platenet.json")
    build_basic_vgg16(config)
