import os
import skimage
import numpy as np

from skimage import io
from skimage import transform

class Reader():
    def __init__(self, directory):
        self.directory = directory

        all_files = os.listdir(self.directory)
        # filter image files
        self.files = filter(
            lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
            all_files)

    def get_samples_list(self):
        ''' Gets list of image and labels filenames.
        '''
        # get labels directory
        for root, dirs, files in os.walk(self.directory):
            files = filter(
                lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
                files)
            for file in files:
                filepath = os.path.join(root, file)
                name = file[0:file.rfind(".")]
                xmlfilepath = os.path.join(root, name + ".xml")

                if os.path.exists(xmlfilepath):
                    # TODO: Task1: парсим xml файл. Если HumanChecked == false, то отбрасываем картинку и xml файл.

                    pass

        # TODO: Task2: Записать пары [image, xmlfile] в samples.
        samples = []

        return samples

    @staticmethod
    def read_batch(img_files, label_files):
        images = []
        labels = []

        for i in range(len(img_files)):
            image = io.imread(img_files[i])
            # TODO: Task3: Придумать как парсить координаты номеров (Reader.parse_label): прямо из xml или использовать сохраненные txt файлы.
            label = Reader.parse_label(label_files[i])

            images.append(image)
            labels.append(label)

        return images, labels


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
                        :return: ndarray of [xmin ymin xmax ymax class]
        '''
        labels = np.loadtxt(path, dtype=int)

        return labels

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
