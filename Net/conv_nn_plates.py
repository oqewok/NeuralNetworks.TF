import tensorflow as tf

IMG_WIDTH = 640
IMG_HEIGHT = 480
CHANNELS = 3
OutputClasses = 2
OutputNodesCount = 4


n_hidden_1 = 96  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_hidden_3 = 512
n_hidden_4 = 1024
n_hidden_5 = 1024
n_hidden_6 = 3072
n_fc_1 = 4096
n_fc_2 = 1000

conv_kernels = {
    'h_conv1': {'size': (11, 11), 'stride': (4, 4)},
    'h_conv2': {'size': (5, 5), 'stride': (2, 2)},
    'h_conv3': {'size': (3, 3), 'stride': (1, 1)},
    'h_conv4': {'size': (3, 3), 'stride': (1, 1)},
    'h_conv5': {'size': (3, 3), 'stride': (1, 1)},
    'h_conv6': {'size': (7, 10), 'stride': (1, 1)},
}

pool_kernels = {
    'h_pool1': {'size': (2, 2), 'stride': (2, 2)},
    'h_pool2': {'size': (2, 2), 'stride': (2, 2)},
    'h_pool5': {'size': (2, 2), 'stride': (2, 2)},
}


# функция инициализации весов
def weight_variable(shape, name='weights'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


weights = {
    'h1': weight_variable(
        [conv_kernels['h_conv1']['size'][0], conv_kernels['h_conv1']['size'][1], 1 * CHANNELS, n_hidden_1], 'W_conv1'),
    'h2': weight_variable(
        [conv_kernels['h_conv2']['size'][0], conv_kernels['h_conv2']['size'][1], n_hidden_1, n_hidden_2], 'W_conv2'),
    'h3': weight_variable(
        [conv_kernels['h_conv3']['size'][0], conv_kernels['h_conv3']['size'][1], n_hidden_2, n_hidden_3], 'W_conv3'),
    'h4': weight_variable(
        [conv_kernels['h_conv4']['size'][0], conv_kernels['h_conv4']['size'][1], n_hidden_3, n_hidden_4], 'W_conv2'),
    'h5': weight_variable(
        [conv_kernels['h_conv5']['size'][0], conv_kernels['h_conv5']['size'][1], n_hidden_4, n_hidden_5], 'W_conv2'),
    'h6': weight_variable(
        [conv_kernels['h_conv6']['size'][0], conv_kernels['h_conv6']['size'][1], n_hidden_5, n_hidden_6], 'W_conv6'),
    'fc1': weight_variable([1 * 1 * n_hidden_6, n_fc_1], 'W_fc1'),
    'fc2': weight_variable([n_fc_1, n_fc_2], 'W_fc2'),
    'softmax': weight_variable([n_fc_2, OutputClasses], 'W_fc2'),
}

biases = {
    'b_conv1': bias_variable([n_hidden_1], 'b_conv1'),
    'b_conv2': bias_variable([n_hidden_2], 'b_conv2'),
    'b_conv3': bias_variable([n_hidden_3], 'b_conv3'),
    'b_conv4': bias_variable([n_hidden_4], 'b_conv4'),
    'b_conv5': bias_variable([n_hidden_5], 'b_conv5'),
    'b_conv6': bias_variable([n_hidden_6], 'b_conv6'),
    'b_fc1': bias_variable([n_fc_1], 'b_conv1'),
    'b_fc2': bias_variable([n_fc_2], 'b_conv2'),
    'softmax': bias_variable([OutputClasses], 'b_fc2'),
}


# функция свертки
def conv2d(x, W, stride=(1, 1), pad_w=1, pad_h=1, padding='SAME', name='conv'):
    return tf.nn.conv2d(x, W, strides=[pad_w, stride[0], stride[1], pad_h],
                        padding=padding, name=name)


# функция пуллинга с функц. максимума.
def max_pool(x, ksize=(2, 2), stride=(2, 2), name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME', name=name)


def conv_layers():
    # Входные изображения x состоят из 2d-тензора чисел с плавающей запятой 480x640x3.
    x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='inputs')

    # First layer (96 kernel conv 11x11, max_pool_2x2). Result: 59x79x96
    W_conv1 = weights['h1']
    b_conv1 = biases['b_conv1']

    h_conv1 = tf.nn.relu(tf.add(
        conv2d(x, W_conv1, stride=conv_kernels['h_conv1']['stride'], padding='VALID', name='h_conv1'),
        b_conv1, name='h_conv1_b'), name='h_conv1_relu')
    h_pool1 = max_pool(
        h_conv1, ksize=pool_kernels['h_pool1']['size'], stride=pool_kernels['h_pool1']['stride'], name='max_pool1')

    # Second layer (256 kernels conv 5x5, max_pool_2x2). Result: 19x14x256
    W_conv2 = weights['h2']
    b_conv2 = biases['b_conv2']

    h_conv2 = tf.nn.relu(tf.add(
        conv2d(h_pool1, W_conv2, stride=conv_kernels['h_conv2']['stride'], padding='VALID', name='h_conv2'), b_conv2,
        name='h_conv2_b'), name='h_conv2_relu')
    h_pool2 = max_pool(
        h_conv2, ksize=pool_kernels['h_pool2']['size'], stride=pool_kernels['h_pool2']['stride'], name='max_pool2')

    # Third layer (512 kernels conv 3x3). Result: 19x14x512
    W_conv3 = weights['h3']
    b_conv3 = biases['b_conv3']

    h_conv3 = tf.nn.relu(tf.add(
        conv2d(h_pool2, W_conv3, stride=conv_kernels['h_conv3']['stride'], name='h_conv3'), b_conv3,
        name="h_conv3_b"), name='h_conv3_relu')

    # Fourth layer (1024 kernels conv 3x3). Result: 19x14x1024
    W_conv4 = weights['h4']
    b_conv4 = biases['b_conv4']

    h_conv4 = tf.nn.relu(tf.add(
        conv2d(h_conv3, W_conv4, stride=conv_kernels['h_conv4']['stride'], name='h_conv4'), b_conv4,
        name="h_conv4_b"), name='h_conv4_relu')

    # Fifth layer (1024 kernels conv_2x2, max_pool_2x2). Result: 5x4x512
    W_conv5 = weights['h5']
    b_conv5 = biases['b_conv5']

    h_conv5 = tf.nn.relu(tf.add(
        conv2d(h_conv4, W_conv5, stride=conv_kernels['h_conv5']['stride'], name='h_conv5'), b_conv5,
        name="h_conv5_b"), name='h_conv5_relu')
    h_pool5 = max_pool(h_conv5, ksize=pool_kernels['h_pool5']['size'], stride=pool_kernels['h_pool5']['stride'],
                       name='max_pool5')

    # Sixth layer (3072 kernels conv 6x6). Result: 19x14x1024
    W_conv6 = weights['h6']
    b_conv6 = biases['b_conv6']

    h_conv6 = tf.nn.relu(tf.add(
        conv2d(h_pool5, W_conv6, stride=conv_kernels['h_conv6']['stride'], padding='VALID', name='h_conv6'), b_conv6, name="h_conv6_b"),
        name='h_conv6_relu')

    return x, h_conv6  # , [W_conv1, b_conv1,
    #  W_conv2, b_conv2,
    #  W_conv3, b_conv3]


def get_training_model():
    x, conv_layer = conv_layers()

    # TODO: прочитать про SGD, как обучить корректировать координаты рамок.
    # Fully connected layer 1
    W_fc1 = weights['fc1']
    b_fc1 = biases['b_fc1']

    conv_layer_flat = tf.reshape(conv_layer, [-1, 1 * 1 * n_hidden_6])
    h_fc1 = tf.add(tf.matmul(conv_layer_flat, W_fc1, name='fc1'), b_fc1, name='fc1_b')

    # Dropout layer.
    fc1_dropout_prob = tf.placeholder(tf.float32, name="fc1_dropout")
    h_fc1_drop = tf.nn.dropout(h_fc1, fc1_dropout_prob)

    # Fully connected layer 2
    W_fc2 = weights['fc2']
    b_fc2 = biases['b_fc2']

    h_fc2 = tf.add(tf.matmul(h_fc1_drop, W_fc2, name='fc2'), b_fc2, name="fc2_b")

    # Softmax layer
    W_softmax = weights['softmax']
    b_softmax = biases['softmax']

    y = tf.nn.softmax(tf.add(tf.matmul(h_fc2, W_softmax, name='softmax'), b_softmax, name="softmax_b"), name="outputs")

    # l2_loss = tf.nn.l2_loss(y)
    return x, y, fc1_dropout_prob,  # l2_loss  # , [conv_vars, W_fc1, b_fc1, W_fc2, b_fc2]
