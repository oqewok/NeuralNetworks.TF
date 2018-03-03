import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class MnistDataProvider():
    def __init__(self, config):
        self.config = config
        # load data here
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def next_batch(self):
        batch = self.mnist.train.next_batch(self.config.batch_size)

        yield batch[0], batch[1]