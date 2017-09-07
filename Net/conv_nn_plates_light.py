import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 64
CHANNELS = 3
OutputClasses = 2
OutputNodesCount = 4


n_hidden_1 = 48  # 1st layer number of features
n_hidden_2 = 64  # 2nd layer number of features
n_hidden_3 = 128

n_fc_1 = 2048

conv_kernels = {
    'h_conv1': {'size': (5, 5), 'stride': (1, 1)},
    'h_conv2': {'size': (5, 5), 'stride': (1, 1)},
    'h_conv3': {'size': (5, 5), 'stride': (1, 1)},
}

pool_kernels = {
    'h_pool1': {'size': (2, 2), 'stride': (2, 2)},
    'h_pool2': {'size': (2, 1), 'stride': (2, 1)},
    'h_pool3': {'size': (2, 2), 'stride': (2, 2)},
}


# функция инициализации весов
def weight_variable(shape, name='weights'):
    initial = tf.random_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape, name='bias'):
    initial = tf.random_normal(shape=shape, name=name)
    return tf.Variable(initial)


weights = {
    'h1': weight_variable(
        [conv_kernels['h_conv1']['size'][0], conv_kernels['h_conv1']['size'][1], 1 * CHANNELS, n_hidden_1], 'W_conv1'),
    'h2': weight_variable(
        [conv_kernels['h_conv2']['size'][0], conv_kernels['h_conv2']['size'][1], n_hidden_1, n_hidden_2], 'W_conv2'),
    'h3': weight_variable(
        [conv_kernels['h_conv3']['size'][0], conv_kernels['h_conv3']['size'][1], n_hidden_2, n_hidden_3], 'W_conv3'),
    'W_fc1_c': weight_variable([32 * 8 * n_hidden_3, n_fc_1], 'W_fc1_c'),
    'sigmoid': weight_variable([n_fc_1, OutputClasses], 'W_sigmoid'),
}

biases = {
    'b_conv1': bias_variable([n_hidden_1], 'b_conv1'),
    'b_conv2': bias_variable([n_hidden_2], 'b_conv2'),
    'b_conv3': bias_variable([n_hidden_3], 'b_conv3'),
    'b_fc1_c': bias_variable([n_fc_1], 'b_fc1_c'),
    'sigmoid_b': bias_variable([OutputClasses], 'sigmoid_b'),
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

    # First layer. Result: 32x64x48
    W_conv1 = weights['h1']
    b_conv1 = biases['b_conv1']

    h_conv1 = tf.nn.relu(tf.add(
        conv2d(x, W_conv1, stride=conv_kernels['h_conv1']['stride'], name='h_conv1'),
        b_conv1, name='h_conv1_b'), name='h_conv1_relu')
    h_pool1 = max_pool(
        h_conv1, ksize=pool_kernels['h_pool1']['size'], stride=pool_kernels['h_pool1']['stride'], name='max_pool1')

    # Second layer. Result: 16x64x64
    W_conv2 = weights['h2']
    b_conv2 = biases['b_conv2']

    h_conv2 = tf.nn.relu(tf.add(
        conv2d(h_pool1, W_conv2, stride=conv_kernels['h_conv2']['stride'], name='h_conv2'), b_conv2,
        name='h_conv2_b'), name='h_conv2_relu')
    h_pool2 = max_pool(
        h_conv2, ksize=pool_kernels['h_pool2']['size'], stride=pool_kernels['h_pool2']['stride'], name='max_pool2')

    # Third layer. Result: 8x32x128
    W_conv3 = weights['h3']
    b_conv3 = biases['b_conv3']

    h_conv3 = tf.nn.relu(tf.add(
        conv2d(h_pool2, W_conv3, stride=conv_kernels['h_conv3']['stride'], name='h_conv3'), b_conv3,
        name="h_conv3_b"), name='h_conv3_relu')
    h_pool3 = max_pool(
        h_conv3, ksize=pool_kernels['h_pool3']['size'], stride=pool_kernels['h_pool3']['stride'], name='max_pool3')

    return x, h_pool3


def get_classification_model():
    x, conv_layer = conv_layers()

    # Fully connected layer 1
    W_fc1 = weights['W_fc1_c']
    b_fc1 = biases['b_fc1_c']

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * n_hidden_3])
    h_fc1 = tf.nn.relu(
        tf.add(tf.matmul(conv_layer_flat, W_fc1, name='fc1_c'), b_fc1, name='fc1_c_b'), name='fc1_relu')

    # Dropout layer.
    fc1_c_dropout_prob = tf.placeholder(tf.float32, name="fc1_c_dropout")
    h_fc1_c_drop = tf.nn.dropout(h_fc1, fc1_c_dropout_prob)

    # sigmoid layer
    W_sigmoid = weights['sigmoid']
    b_sigmoid = biases['sigmoid_b']

    y = tf.add(tf.matmul(h_fc1_c_drop, W_sigmoid, name='sigmoid'), b_sigmoid, name="sigmoid_b")
    return x, y, fc1_c_dropout_prob,
