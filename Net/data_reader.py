import tensorflow as tf
import numpy as np
import image_proc

IMG_WIDTH = 128
IMG_HEIGHT = 96

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train.txt'

# читает .txt файл, в котором в каждой строке через пробел отделены путь к файлу с изображением и путь к разметке
# изображения.
def read_labeled_image_list():
    f = open(TRAIN_FILE_PATH, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(label)

    f.close()
    return filenames, labels

# читает изображения и их маски из файлов
def read_images_from_disk(image_files, masks_file_list):
    masks = read_masks(masks_file_list)
    images = []

    for image_file in image_files:
        #file_contents = tf.read_file(image_file)
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        image = image_proc.read_and_normalize(image_file)
        #t = tf.convert_to_tensor(image, dtype=tf.float32)
        images.append(image)
    #tf.image.resize_bicubic(images, [IMG_HEIGHT, IMG_WIDTH])
    return [images, masks]


# читает из переданных файлов маски изображений и возвращает массив float из этих масок.
def read_masks(masks_file_list):
    masks = []

    for mask_file in masks_file_list:
        with open(mask_file, 'r') as file:
            data = file.read()
            arr = np.array(data.split(','), float)
            #arr = tf.convert_to_tensor(arr, dtype=tf.float32)
            masks.append(arr)

    return masks


# читаем пути к файлам с изображениями и их масками
# names, labels = read_labeled_image_list()
# images, masks = read_images_from_disk(names, labels)
