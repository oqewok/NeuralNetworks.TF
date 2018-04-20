import numpy as np
import os
import math

from  Structured.utils.img_preproc import *
from Structured.data_loader.reader import Reader
from Structured.utils.config import process_config

class CarPlatesDataProvider():
    def __init__(self, config):
        self.config = config

        # define train, valid and test samples
        # Заранее копируем из ЯД в локальные папки Train, Test, Valid выборку с разметкой.
        self.samples = {
            'TRAIN': CarPlatesDataProvider.getSamplesFilenames(os.path.abspath(
        os.path.join(config.train_files_directory, "train.txt"))),
            'VALID': CarPlatesDataProvider.getSamplesFilenames(os.path.abspath(
        os.path.join(config.train_files_directory, "valid.txt"))),
            'TEST' : CarPlatesDataProvider.getSamplesFilenames(os.path.abspath(
        os.path.join(config.train_files_directory, "test.txt"))),
        }

        self.num_train = len(self.samples['TRAIN'][0])
        self.num_valid = len(self.samples['VALID'][0])
        self.num_test  = len(self.samples['TEST' ][0])

        self.batch_idx = 0
        self.batch_size = config.batch_size
        self.num_batches = int(math.ceil(self.num_train / float(self.batch_size)))

        self.order = np.arange(self.num_train)

        self.X_val, self.Y_val = self.samples['VALID'][0], self.samples['VALID'][1]


    # Загружает данные о разметке из txt-файлов в память.
    @staticmethod
    def getSamplesFilenames(directory):
        '''   @:return: list of [[img_file1, label_file1], [img_file2, label_file2], ...]
        '''
        f = open(directory, 'r', encoding='utf-8')
        imgs, labels = [], []

        for line in f:
            try:
                image_file, label_file = line[:-1].split('  ')

                image_file = os.path.abspath(image_file)
                label_file = os.path.abspath(label_file)

                imgs.append(image_file)
                labels.append(label_file)
            except ValueError:
                print(directory)
                print(line)
                raise ValueError

        f.close()
        return np.array(imgs), np.array(labels)

    def next_batch(self, batch_size=1, type="TRAIN"):
        ''' Reads the batch of images and bboxes
        '''

        img_files = self.samples[type][0]
        bboxes_files = self.samples[type][1]

        indices = None
        if type == "TRAIN":
            indices = self.order[self.batch_idx * batch_size:self.batch_idx * batch_size + batch_size]

            if self.batch_idx < self.num_batches - 1:
                self.batch_idx = self.batch_idx + 1
            else:
                self.order = np.random.permutation(self.num_train)
                self.batch_idx = 0

        elif type == "TEST":
            indices = np.random.permutation(self.num_test)
        else:
            indices = np.random.permutation(self.num_valid)

        img_files = img_files[indices]
        bboxes_files = bboxes_files[indices]

        self.images, self.bboxes = Reader.read_batch(img_files, bboxes_files, new_shape=self.config.input_shape)

        return self.images, self.bboxes


    def next_img(self):
        assert self.batch_size == 1

        images, bboxes = self.next_batch()
        self.images, self.bboxes = images[0], bboxes[0]

        return self.images, self.bboxes
'''
    @staticmethod
    def load_imgs(img_files):
        imgs = Reader.read_imgs(img_files)

        return imgs
'''

"""
config = process_config("E:/Study/Mallenom/NeuralNetworks.TF/Net/Structured/configs/fastercnn.json")
data_provider = CarPlatesDataProvider(config)

batches = []

images, bboxes = data_provider.next_img()
pass
"""

'''
Reader.get_samples_file("E:/YandexDisk/testsamples/frames/Абхазия(AB)/", "E:/train.txt")
samples = CarPlatesDataProvider.getSamplesFilenames("E:/train.txt")

pass
'''