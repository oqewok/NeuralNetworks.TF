import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

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
sess.run(tf.global_variables_initializer())

# Предсказанная функция класса и потерь (регрессионая модель)
y = tf.matmul(x, weights) + biases

# Функция потерь (loss). Функция потерь - это кросс-энтропия между мишенью и функцией активации softmax, применяемой
# к предсказанию модели. Потеря показывает, насколько плохим было предсказание модели на одном примере.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Оптимизация методом крутого градиентного спуска с шагом 0,5. Операция train_step при запуске будет применять
# обновления спуска градиента к параметрам. Поэтому обучение модели может быть выполнено путем многократного запуска
# train_step.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# На каждой тренировочной итерации мы загружаем 100 примеров обучения. Затем мы запускаем train_step операцию,
# используя feed_dict для замены placeholder тензоров x и y_примеры обучения. Обратите внимание, что вы можете заменить
# любой тензор на вашем графике вычислений, используя feed_dict - это не ограничивается только placeholders.
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Оценка модели.
# tf.argmax является чрезвычайно полезной функцией, которая дает вам индекс наивысшей записи в тензоре
# вдоль некоторой оси. Например, tf.argmax(y,1) ярлык, который, по мнению нашей модели, наиболее вероятен для каждого
# входа, а tf.argmax(y_,1) является истинной меткой. Мы можем использовать, tf.equal чтобы проверить, соответствует ли
# наше предсказание истине.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
