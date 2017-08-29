import tensorflow as tf

IMG_WIDTH = 640
IMG_HEIGHT = 480
CHANNELS = 3
OutputNodesCount = 4


# функция инициализации весов
def weight_variable(shape, name='weights'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# функция свертки
def conv2d(x, W, stride=(1, 1), pad_w=1, pad_h=1, padding='SAME', name='conv'):
    return tf.nn.conv2d(x, W, strides=[pad_w, stride[0], stride[1], pad_h],
                        padding=padding, name=name)


# функция пуллинга с функц. максимума.
def max_pool(x, ksize=(2, 2), stride=(2, 2), name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME', name=name)


def conv_layers():
    # Входные изображения x состоят из 2d-тензора чисел с плавающей запятой.
    x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='inputs')

    # First layer (48 kernel conv_7x7, max_pool_2x2). Result: 160x120x64
    W_conv1 = weight_variable([7, 7, 1 * CHANNELS, 48], 'W_conv1')
    b_conv1 = bias_variable([48], 'b_conv1')

    h_conv1 = tf.nn.relu(tf.add(
        conv2d(x, W_conv1, stride=(2, 2), name='h_conv1'), b_conv1, name='h_conv1_b'), name='h_conv1_relu')
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer (64 kernels conv_5x5, max_pool_2x2). Result: 40x30x64
    W_conv2 = weight_variable([5, 5, 48, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')

    h_conv2 = tf.nn.relu(tf.add(
        conv2d(h_pool1, W_conv2, stride=(2, 2), name='h_conv2'), b_conv2, name='h_conv2_b'), name='h_conv2_relu')
    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))

    # Third layer (128 kernels conv_5x5, max_pool_2x2). Result: 20x15x128
    W_conv3 = weight_variable([3, 3, 64, 128], 'W_conv2')
    b_conv3 = bias_variable([128], 'b_conv2')

    h_conv3 = tf.nn.relu(tf.add(
        conv2d(h_pool2, W_conv3, stride=(1, 1), name='h_conv3'), b_conv3, name="h_conv3_b"), name='h_conv3_relu')
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2), name='max_pool3')

    # Dropout
    h_conv3_dropout_prob = tf.placeholder(tf.float32, name="h_conv3_dropout_prob")
    h_conv3_drop = tf.nn.dropout(h_pool3, h_conv3_dropout_prob, name='h_conv3_drop')

    # Fourth layer (256 kernels conv_2x2, max_pool_2x2). Result: 10x8x256
    W_conv4 = weight_variable([2, 2, 128, 256], 'W_conv2')
    b_conv4 = bias_variable([256], 'b_conv2')

    h_conv4 = tf.nn.relu(tf.add(
        conv2d(h_conv3_drop, W_conv4, stride=(1, 1), name='h_conv4'), b_conv4, name="h_conv4_b"), name='h_conv4_relu')
    h_pool4 = max_pool(h_conv4, ksize=(2, 2), stride=(2, 2), name='max_pool4')

    # Fifth layer (512 kernels conv_2x2, max_pool_2x2). Result: 5x4x512
    W_conv5 = weight_variable([2, 2, 256, 512], 'W_conv2')
    b_conv5 = bias_variable([512], 'b_conv2')

    h_conv5 = tf.nn.relu(tf.add(
        conv2d(h_pool4, W_conv5, stride=(1, 1), name='h_conv5'), b_conv5, name="h_conv5_b"), name='h_conv5_relu')
    h_pool5 = max_pool(h_conv5, ksize=(2, 2), stride=(2, 2), name='max_pool5')

    return x, h_pool5, h_conv3_dropout_prob   # , [W_conv1, b_conv1,
                                              #  W_conv2, b_conv2,
                                              #  W_conv3, b_conv3]


def get_training_model():
    x, conv_layer, h_conv3_dropout_prob = conv_layers()

    # TODO: прочитать про SGD, как обучить корректировать координаты рамок.
    # Fully connected layer
    W_fc1 = weight_variable([5 * 4 * 512, 1024], 'W_fc1')
    b_fc1 = bias_variable([1024], 'b_fc1')

    conv_layer_flat = tf.reshape(conv_layer, [-1, 5 * 4 * 512])
    h_fc1 = tf.add(tf.matmul(conv_layer_flat, W_fc1, name='fc1'), b_fc1, name='fc1_b')

    # дополнительный параметр fc1_dropout в системе feed_dict для управления отсевом.
    fc1_dropout_prob = tf.placeholder(tf.float32, name="fc1_dropout")
    h_fc1_drop = tf.nn.dropout(h_fc1, fc1_dropout_prob)

    # Output layer
    W_fc2 = weight_variable([1024, OutputNodesCount], 'W_fc2')
    b_fc2 = bias_variable([OutputNodesCount], 'b_fc2')

    y = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="outputs")

    l2_loss = tf.nn.l2_loss(y)
    return x, y, fc1_dropout_prob, h_conv3_dropout_prob, l2_loss  # , [conv_vars, W_fc1, b_fc1, W_fc2, b_fc2]
