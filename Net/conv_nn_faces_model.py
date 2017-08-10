import tensorflow as tf


# функция инициализации весов
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# функция свертки
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# функция пуллинга с функц. максимума.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Входные изображения x состоят из 2d-тензора чисел с плавающей запятой. Здесь мы изображаем его форму как [None,
# 128 * 96], где 128 * 96 это размерность один сведенный 128 на 96 пикселя зображения, и None указывает на то,
# что первое измерение, соответствующее размера партии, может быть любого размера.
x = tf.placeholder(tf.float32, shape=[None, 128 * 96])
# Целевые классы результатов y_ также будут состоять из 2d-тензора, где каждая строка является одним 255-мерным
# вектором, указывающим, в какой области вероятнее всего находится лицо.
y_ = tf.placeholder(tf.float32, shape=[None, 225])


filename_queue = tf.train.input_producer(['E:/data/gt_db/gt_db.zip'])

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

