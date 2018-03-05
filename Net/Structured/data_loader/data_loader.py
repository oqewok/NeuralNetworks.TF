import numpy as np
import os
import math

from Structured.data_loader.reader import Reader

class DataLoader():
    def __init__(self, config):
        self.config = config

        # define train, valid and test samples
        # TODO: Task4: Разобраться, где будут лежать train, valid и test выборка. В root_directory предполагается train.
        # Заранее копируем из ЯД в локальные папки Train, Test, Valid выборку с разметкой.
        self.samples = {
            'TRAIN': DataLoader.getSamplesFilenames(os.path.abspath(config.train_root_directory)),
            'VALID': DataLoader.getSamplesFilenames(os.path.abspath(config.valid_root_directory)),
            'TEST' : DataLoader.getSamplesFilenames(os.path.abspath(config.test_root_directory)),
        }

        self.num_train = len(self.samples['TRAIN'])
        self.num_valid = len(self.samples['VALID'])
        self.num_test  = len(self.samples['TEST'])

        self.batch_idx = 0
        self.batch_size = config.batch_size
        self.num_batches = int(math.ceil(self.num_train / float(self.batch_size)))

        self.order = np.random.permutation(self.num_train)


    # Загружает данные о разметке из txt-файлов в память.
    @staticmethod
    def getSamplesFilenames(directory):
        '''   @:return: list of [[img_file1, label_file1], [img_file2, label_file2], ...]
        '''
        f = open(directory, 'r')
        samples = []

        for line in f:
            try:
                image_file, label_file = line[:-1].split('  ')
                samples.append([image_file, label_file])
            except ValueError:
                print(directory)
                print(line)
                raise ValueError

        f.close()
        return samples


    def next_batch(self):
        ''' Reads the batch of images and labels
                @:param batch_size: length of image batch
                @:param order:      array of shuffled indices size of self.num_train
        '''

        img_files = self.samples['TRAIN'][0]
        label_files = self.samples['TRAIN'][1]

        indices = self.order[self.batch_idx * self.batch_size:self.batch_idx * self.batch_size + self.batch_size]

        if self.batch_idx < self.num_batches - 1:
            self.batch_idx = self.batch_idx + 1
        else:
            self.order = np.random.permutation(self.num_train)
            self.batch_idx = 0

        img_files = img_files[indices]
        label_files = label_files[indices]

        images, labels = Reader.read_batch(img_files, label_files)

        return images, labels


    @staticmethod
    def load_imgs(img_files):
        imgs = Reader.read_imgs(img_files)

        return imgs

'''
Reader.get_samples_file("E:/YandexDisk/testsamples/frames/Абхазия(AB)/", "E:/train.txt")
samples = DataLoader.getSamplesFilenames("E:/train.txt")

pass
'''