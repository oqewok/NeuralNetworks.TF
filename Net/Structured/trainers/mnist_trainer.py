from Structured.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class MnistTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(MnistTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for i in loop:
            loss, acc = self.train_step()
            # Запилить вычисление среднего значения точности по обучающей выборке
            # Запилить сохранение модели после эпохи

        # Пофиксить непонятную ошибку.
        print('test accuracy %g' % self.model.accuracy.eval(feed_dict={self.model.x: self.data.mnist.test.images, self.model.y: self.data.mnist.test.labels, self.model.keep_prob: 1.0}))

        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """

        # from template or example code
        # batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        # feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        # _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
        #                              feed_dict=feed_dict)

        b_x, b_y = next(self.data.next_batch()) #?????
        feed_dict = {self.model.x: b_x, self.model.y: b_y, self.model.keep_prob : self.config.dropout}

        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)

        return loss, acc



''''''