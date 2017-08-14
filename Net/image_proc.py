import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io
from skimage import transform
from PIL import Image


IMG_WIDTH = 128
IMG_HEIGHT = 96
PATH = 'E:/data/gt_db/s01/01.jpg'


def normalize(image):
    return image[0:] / 255


def read(path):
    # image = io.imread(path)
    image = Image.open(path)
    return image


def read_and_normalize(path):
#    image = transform.resize(read(path), [IMG_HEIGHT, IMG_WIDTH], mode='reflect')
#   i = np.reshape(image, 480 * 640 * 3)
    image = Image.open(path)
    im = image.resize((128, 96))
    img = np.array(im) / 255
    return img


def show_image(image):
    plt.imshow(image)
    plt.waitforbuttonpress()


#image = read(PATH)
#img = skimage.img_as_float(image)
#im = np.array(img)
#a = np.reshape(img, 480 * 640 * 3)
#img = Image.open(PATH)
#im = img.resize((96, 128))
#i = np.array(im) / 255
#print()
# n_image = normalize(image)
# show_image(n_image)
