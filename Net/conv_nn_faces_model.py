import tensorflow as tf

IMG_WIDTH = 320
IMG_HEIGHT = 240
CHANNELS = 3
OutputNodesCount = 4


# функция инициализации весов
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# функция свертки
def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)


# функция пуллинга с функц. максимума.
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def conv_layers():
    # Входные изображения x состоят из 2d-тензора чисел с плавающей запятой.
    x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='inputs')

    # First layer (1 kernel conv_7x7, max_pool_3x3). Result: 40x54x1
    W_conv1 = weight_variable([7, 7, 1 * CHANNELS, 1], 'W_conv1')
    b_conv1 = bias_variable([1], 'b_conv1')

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=(2, 2)) + b_conv1, name='h_conv1_relu')
    h_pool1 = max_pool(h_conv1, ksize=(3, 3), stride=(3, 3))

    # Second layer (4 kernels conv_5x5, max_pool_2x2). Result: 10x14x8
    W_conv2 = weight_variable([5, 5, 1, 8], 'W_conv2')
    b_conv2 = bias_variable([8], 'b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=(2, 2)) + b_conv2, name='h_conv2_relu')
    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))

    return x, h_pool2  # , [W_conv1, b_conv1,
                       #  W_conv2, b_conv2,
                       #  W_conv3, b_conv3]


def get_training_model():
    x, conv_layer = conv_layers()

    # Fully connected layer
    W_fc1 = weight_variable([10 * 14 * 8, 128], 'W_fc1')
    b_fc1 = bias_variable([128], 'b_fc1')

    conv_layer_flat = tf.reshape(conv_layer, [-1, 10 * 14 * 8])
    h_fc1 = tf.nn.sigmoid(tf.matmul(conv_layer_flat, W_fc1) + b_fc1, name='sigmoid')

    # дополнительный параметр keep_prob в системе feed_dict для управления отсевом.
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([128, OutputNodesCount], 'W_fc2')
    b_fc2 = bias_variable([OutputNodesCount], 'b_fc2')

    y = tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="outputs")

    return x, y, keep_prob  # , [conv_vars, W_fc1, b_fc1, W_fc2, b_fc2]
