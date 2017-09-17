import tensorflow as tf


# функция инициализации весов
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


# функция инициализации байесов.
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# функция свертки
def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


# функция пуллинга с функц. максимума.
def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

# Веса модели
weights = tf.Variable(tf.zeros([784, 10]), name="weights")
# предубеждения
biases = tf.Variable(tf.zeros([10]), name="biases")

# Первый сверточный слой

W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

x_image = tf.reshape(x, [-1, 28, 28, 1], "x_image")

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, "h_conv1_relu") + b_conv1, name="h_conv1")
h_pool1 = max_pool_2x2(h_conv1, "h_pool1")

# Второй сверточный слой.
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_convl2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, "h_conv2_relu") + b_conv2, name="h_conv2")
h_pool2 = max_pool_2x2(h_conv2, name="h_pool2")

# Полносвязный слой.
W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
b_fc1 = bias_variable([1024], name="b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name="h_pool2_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

# Чтобы уменьшить переобучение, мы применим исключение до уровня считывания. Мы создаем placeholder вероятность того,
# что выход нейрона будет сохранен во время отсева. Это позволяет нам отказаться от участия во время обучения и
# отключить его во время тестирования. Оператор TensorFlow tf.nn.dropout автоматически обрабатывает масштабирование
# выходов нейронов в дополнение к их маскировке, поэтому выпадение просто работает без какого-либо дополнительного
# масштабирования.

# дополнительный параметр dropout в системе feed_dict для управления отсевом.
keep_prob = tf.placeholder(tf.float32, name="dropout")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fcl_drop")

# Выходной слой
W_fc2 = weight_variable([1024, 10], name="W_fc2")
b_fc2 = bias_variable([10], name="dropout")

y_conv = tf.matmul(h_fc1_drop, W_fc2, name="y_conv") + b_fc2

# Обучение и оценка модели

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name="cross_entropy")

# Оптимизация алгоритмом ADAM.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")



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


saver = tf.train.Saver()
# write logs and graph for TB
# writer = tf.train.SummaryWriter("/logs", graph=tf.get_default_graph())




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())





    # histogram
    histogram = tf.summary.histogram(name="histogram", values=y_)

    #scalar
    summary_scalar = tf.summary.scalar('loss', accuracy)

    summary_all = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)


    writer = tf.summary.FileWriter("./logs/Summary_FileWriter", sess.graph)
    # writer2 = tf.train.SummaryWriter("./logs/Train_SummaryWriter", graph=tf.get_default_graph())

    for i in range(200):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

            # save model
            saver.save(sess, './models/my-model-step_' + str.format(i.__str__()))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('Saving...')
    saver.save(sess, './models/my-model-step_final')
    print('Saving complete.')

    writer = tf.summary.FileWriter("./logs/Merged", sess.graph);

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, dropout: 1.0}))
