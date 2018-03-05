import numpy as np
import os
import math

from data_loader.reader import Reader

class DataLoader():
    def __init__(self, config):
        self.config = config

        self.root_directory = config.root_directory

        # define train, valid and test samples
        # TODO: Task4: Разобраться, где будут лежать train, valid и test выборка. В root_directory предполагается train.
        # Заранее копируем из ЯД в локальные папки Train, Test, Valid выборку с разметкой.
        self.samples = {
            'TRAIN': DataLoader.read_filenames_list(os.path.abspath(self.root_directory)),
            'VALID': DataLoader.read_filenames_list(os.path.abspath(self.root_directory)),
            'TEST' : DataLoader.read_filenames_list(os.path.abspath(self.root_directory)),
        }

        self.num_train = len(self.samples['TRAIN'])
        self.num_valid = len(self.samples['VALID'])
        self.num_test  = len(self.samples['TEST'])

        self.batch_idx = 0
        self.batch_size = config.batch_size
        self.num_batches = int(math.ceil(self.num_train / float(self.batch_size)))

        self.order = np.random.permutation(self.num_train)

    @staticmethod
    def read_filenames_list(directory):
        ''' Reads list of filenames pairs.
                @:return: list of [[img_file1, label_file1], [img_file2, label_file2], ...]
        '''
        reader = Reader(directory)
        # list of sample = [image, label]
        samples = reader.get_samples_list()

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