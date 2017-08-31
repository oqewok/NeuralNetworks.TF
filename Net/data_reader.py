import conv_nn_plates
import numpy as np
import image_proc

IMG_WIDTH = conv_nn_plates.IMG_WIDTH
IMG_HEIGHT = conv_nn_plates.IMG_HEIGHT

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train.txt'


# читает .txt файл, в котором в каждой строке через пробел отделены путь к файлу с изображением и путь к разметке
# изображения.
def read_labeled_image_list(path):
    f = open(path, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split('  ')
        filenames.append(filename)
        labels.append(label)

    f.close()
    return filenames, labels


# читает изображения и их маски из файлов
def read_images_from_disk(image_files, masks_file_list):
    masks = read_masks(masks_file_list)
    images = []

    for image_file in image_files:
        image = image_proc.read_and_normalize(image_file)
        images.append(image)

    return [images, masks]


# читает изображение из файла
def read_image(image_file):
    image = image_proc.read_and_normalize(image_file)
    return image


# читает из переданных файлов маски изображений и возвращает массив float из этих масок.
def read_masks(masks_file_list):
    masks = []

    for mask_file in masks_file_list:
        with open(mask_file, 'r') as file:
            data = file.read()
            arr = np.array(data.split(','), float)
            masks.append(arr)

    return masks

# читаем пути к файлам с изображениями и их масками
# names, labels = read_labeled_image_list()
# images, masks = read_images_from_disk(names, labels)
