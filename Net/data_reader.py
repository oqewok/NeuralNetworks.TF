import tensorflow as tf
from skimage import io
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 96

# читает .txt файл, в котором в каждой строке через пробел отделены путь к файлу с изображением и путь к разметке
# изображения.
def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(label)

    f.close()
    return filenames, labels


def read_images_from_disk(image_files, masks_file_list):
    masks = read_masks(masks_file_list)
    images = []

    for image_file in image_files:
        file_contents = tf.read_file(image_file)
        image = tf.image.decode_jpeg(file_contents, channels=3)
        images.append(image)

    result_images = tf.image.resize_bicubic(images, [IMG_HEIGHT, IMG_WIDTH])
    return result_images, masks


# читает из переданных файлов маски изображений и возвращает массив float из этих масок.
def read_masks(masks_file_list):
    masks = []

    for mask_file in masks_file_list:
        with open(mask_file, 'r') as file:
            data = file.read()
            arr = np.array(data.split(','), float)
            masks.append(arr)

    return masks


def get_batch(data, count):
    # здесь берем порциями данные
    print()


# читаем пути к файлам с изображениями и их масками
names, labels = read_labeled_image_list('E:/Study/Mallenom/train.txt')

images, masks = read_images_from_disk(names, labels)

print()
