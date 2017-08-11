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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Входные изображения x состоят из 2d-тензора чисел с плавающей запятой. Здесь мы изображаем его форму как [None, 784],
# где 784 это размерность один сведенный 28 на 28 пикселя MNIST изображения, и None указывает на то, что первое
# измерение, соответствующее размера партии, может быть любого размера.
x = tf.placeholder(tf.float32, shape=[None, 784])
# Целевые классы результатов y_ также будут состоять из 2d-тензора, где каждая строка является одним 10-мерным
# вектором, указывающим, какой разрядный класс (от нуля до девяти) соответствует соответствующему изображению MNIST.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Веса модели
weights = tf.Variable(tf.zeros([784, 10]))
# предубеждения
biases = tf.Variable(tf.zeros([10]))

# запуск сессии
# sess.run(tf.global_variables_initializer())

# Первый сверточный слой

# Слой будет состоять из свертки, за которым следует макс. пуллинг. Свертка рассчитает 32 функции для
# каждого ядра 5x5. Его весовой тензор будет иметь форму [5, 5, 1, 32]. Первые два измерения - размер ядра,
# в следующем - количество входных каналов, а последнее - количество выходных каналов. У нас также будет вектор
# смещения с компонентом для каждого выходного канала.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# Чтобы применить слой, мы сначала приведем x к тензору 4d, со вторым и третьим размерами, соответствующими ширине и
# высоте изображения, и окончательному размеру, соответствующему количеству цветных каналов.
x_image = tf.reshape(x, [-1, 28, 28, 1])
# Затем мы свернем x_image с весовым тензором, добавим смещение, применим функцию ReLU и, наконец, максимальный пул.
# max_pool_2x2 Метод позволит уменьшить размер изображения до 14х14.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Второй сверточный слой.

# Чтобы построить глубокую сеть, мы складываем несколько слоев этого типа. Второй слой будет иметь 64 функции для
# каждого ядра 5x5.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Полносвязный слой.

# Теперь, когда размер изображения был уменьшен до 7x7, мы добавляем полностью подключенный слой с 1024 нейронами,
# чтобы разрешить обработку всего изображения. Мы преобразуем тензор из пула слоя в партию векторов, умножим на
# весовую матрицу, добавим смещение и применим ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Чтобы уменьшить переобучение, мы применим исключение до уровня считывания. Мы создаем placeholder вероятность того,
# что выход нейрона будет сохранен во время отсева. Это позволяет нам отказаться от участия во время обучения и
# отключить его во время тестирования. Оператор TensorFlow tf.nn.dropout автоматически обрабатывает масштабирование
# выходов нейронов в дополнение к их маскировке, поэтому выпадение просто работает без какого-либо дополнительного
# масштабирования.

# дополнительный параметр keep_prob в системе feed_dict для управления отсевом.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Выходной слой
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Обучение и оценка модели

# Функция потерь (loss). Функция потерь - это кросс-энтропия между мишенью и функцией активации softmax, применяемой
# к предсказанию модели. Потеря показывает, насколько плохим было предсказание модели на одном примере.
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Оптимизация алгоритмом ADAM.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# tf.argmax является чрезвычайно полезной функцией, которая дает вам индекс наивысшей записи в тензоре
# вдоль некоторой оси. Например, tf.argmax(y,1) ярлык, который, по мнению нашей модели, наиболее вероятен для каждого
# входа, а tf.argmax(y_,1) является истинной меткой. Мы можем использовать, tf.equal чтобы проверить, соответствует ли
# наше предсказание истине.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#
# in env.variables
# CUDA_VISIBLE_DEVICES = 1
#
# in program
# export CUDA_VISIBLE_DEVICES="0"


# 1st probably lack of memory problem solution (OOM-problem)
#   config = tf.ConfigProto()
#   config.gpu_options.allow_growth = True
#   sess = tf.Session(config = config)
#

# 2nd probably lack of memory problem solution (OOM-problem)
#   config = tf.ConfigProto()
#   config.gpu_options.per_process_gpu_memory_fraction = 0.4
#   sess = tf.Session(config=config)

# 3rd probably lack of memory problem solution (OOM-problem)
#   config=tf.ConfigProto()
#   # config.gpu_options.per_process_gpu_memory_fraction=0.98
#   config.gpu_options.allocator_type="BFC"
#   config.log_device_placement=True
#   sess=tf.Session(config=config)


# # 1st trial solution
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # line from 3rd
# # config.gpu_options.allocator_type="BFC"
# sess = tf.Session(config = config)


# 2nd trial solution
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# sess = tf.Session(config=config)


# # compiled 1+2+3 solution
# config=tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction=0.98
# config.gpu_options.allocator_type="BFC"
# #config.log_device_placement=True
# sess=tf.Session(config=config)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
