import os
import numpy as np

from skimage import io
from Structured.data_loader.parser import MarkupParser

class Reader():
    def __init__(self, directory):
        self.directory = directory

        all_files = os.listdir(self.directory)
        # filter image files
        self.files = filter(
            lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
            all_files)

    @staticmethod
    def get_samples_file(directory, samples_file):
        ''' Gets list of image and labels filenames.
        '''
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Get xml-parser
        parser = MarkupParser()
        with open(samples_file, "w") as result_file:
            # get labels directory
            for root, dirs, files in os.walk(directory):
                files = filter(
                    lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
                    files)
                for file in files:
                    filepath = os.path.abspath(
                        os.path.join(root, file))

                    name = file[0:file.rfind(".")]

                    xmlfilepath = os.path.abspath(os.path.join(
                        root, name + ".xml"))

                    if os.path.exists(xmlfilepath):
                        # парсим xml файл. Если HumanChecked == false, то отбрасываем картинку и xml файл.
                        if parser.getHumanCheckedValueAttr(xmlfilepath):
                            result_file.write(filepath + "  " + xmlfilepath + "\n")

    @staticmethod
    def read_batch(img_files, label_files):
        images = []
        labels = []

        for img_file in img_files:
            image = io.imread(img_file)
            images.append(image)

        for label_file in label_files:
            # Парсим координаты номеров прямо из xml файла.
            label = Reader.parse_label(label_file)

            labels.append(label)

        return np.array(images), np.array(labels)


    @staticmethod
    def read_imgs(img_files):
        images = []

        for i in range(len(img_files)):
            image = io.imread(img_files[i])
            images.append(image)

        return images


    @staticmethod
    def parse_label(path):
        ''' Parse label file. Using whitespace delimeter.
                        :param path: path to label file
                        :return: ndarray of [[xmin1 ymin1 xmax1 ymax1], ...,[xminN yminN xmaxN ymaxN]]
        '''
        # Get xml-parser
        parser = MarkupParser()
        labels = parser.getLabels(path)

        return labels


'''
Reader.get_samples_file("E:/YandexDisk/testsamples/frames/Абхазия(AB)/", "E:/train.txt")
'''
'''
for root, dirs, files in os.walk("E:/Study/Спецификация, архитектура и проектирование ПО"):
    for file in files:
        print(os.path.join(root, file))
'''

'''
directory = 'E:/data/Финляндия(FI)/'
all_files = os.listdir(directory)

os.chdir(directory)
os.chdir("..")
os.chdir("..")
os.chdir("./Study/")
labels_dir = os.path.abspath(os.getcwd())

# filter image files
files = filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
               all_files)

for file in files:
    name = file[0:file.rfind(".")]
    image = os.path.abspath(directory + '/' + file)
    label = os.path.abspath(labels_dir + '/' + name + ".txt")
    pass
'''