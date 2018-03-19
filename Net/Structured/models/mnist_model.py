from base.base_model import BaseModel
from utils.operations import *

import tensorflow as tf


class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)

        self.lr = config.learning_rate

        self.build_model()
        self.init_saver()




    def build_model(self):
        # here you predict the tensorflow graph of any model you want and also define the loss.

        # Входные изображения x состоят из 2d-тензора чисел с плавающей запятой. Здесь мы изображаем его форму как [None, 784],
        # где 784 это размерность один сведенный 28 на 28 пикселя MNIST изображения, и None указывает на то, что первое
        # измерение, соответствующее размера партии, может быть любого размера.
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        # Целевые классы результатов y_ также будут состоять из 2d-тензора, где каждая строка является одним 10-мерным
        # вектором, указывающим, какой разрядный класс (от нуля до девяти) соответствует соответствующему изображению MNIST.
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # Первый сверточный слой

        # Слой будет состоять из свертки, за которым следует макс. пуллинг. Свертка рассчитает 32 функции для
        # каждого ядра 5x5. Его весовой тензор будет иметь форму [5, 5, 1, 32]. Первые два измерения - размер ядра,
        # в следующем - количество входных каналов, а последнее - количество выходных каналов. У нас также будет вектор
        # смещения с компонентом для каждого выходного канала.
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # Чтобы применить слой, мы сначала приведем x к тензору 4d, со вторым и третьим размерами, соответствующими ширине и
        # высоте изображения, и окончательному размеру, соответствующему количеству цветных каналов.
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
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

        # дополнительный параметр dropout в системе feed_dict для управления отсевом.
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Выходной слой
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Обучение и оценка модели

        # Функция потерь (loss). Функция потерь - это кросс-энтропия между мишенью и функцией активации softmax, применяемой
        # к предсказанию модели. Потеря показывает, насколько плохим было предсказание модели на одном примере.
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv))
        # Оптимизация алгоритмом ADAM.
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # tf.argmax является чрезвычайно полезной функцией, которая дает вам индекс наивысшей записи в тензоре
        # вдоль некоторой оси. Например, tf.argmax(y,1) ярлык, который, по мнению нашей модели, наиболее вероятен для каждого
        # входа, а tf.argmax(y_,1) является истинной меткой. Мы можем использовать, tf.equal чтобы проверить, соответствует ли
        # наше предсказание истине.
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass
