import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 96
CHANNELS = 3
OutputNodesCount = 225


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
    x = tf.placeholder(tf.float32, shape=[None, 96, 128, 3], name='inputs')

    # First layer (conv_5x5, max_pool_2x2). Result: 64x48x48
    W_conv1 = weight_variable([5, 5, 1 * CHANNELS, 48], 'W_conv1')
    b_conv1 = bias_variable([48], 'b_conv1')
    # x_expanded = tf.expand_dims(x, 3)

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1, name='h_conv1_relu')
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer (conv_5x5, max_pool_1x2). Result: 64x24x64
    W_conv2 = weight_variable([5, 5, 48, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2_relu')
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # Third layer (conv_5x5, max_pool_2x2). Result: 32x12x128
    W_conv3 = weight_variable([5, 5, 64, 128], 'W_conv3')
    b_conv3 = bias_variable([128], 'b_conv3')

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name='h_conv3_relu')
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3  # , [W_conv1, b_conv1,
                       #  W_conv2, b_conv2,
                       #  W_conv3, b_conv3]


def get_training_model():
    x, conv_layer = conv_layers()

    # Fully connected layer
    W_fc1 = weight_variable([32 * 12 * 128, 4096], 'W_fc1')
    b_fc1 = bias_variable([4096], 'b_fc1')

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 12 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1, name='fc1_relu')

    # дополнительный параметр keep_prob в системе feed_dict для управления отсевом.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([4096, OutputNodesCount], 'W_fc2')
    b_fc2 = bias_variable([OutputNodesCount], 'b_fc2')

    y = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_sigmoid')

    return x, y, keep_prob  # , [conv_vars, W_fc1, b_fc1, W_fc2, b_fc2]
