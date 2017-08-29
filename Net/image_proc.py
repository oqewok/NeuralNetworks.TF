import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io
from skimage import transform
import conv_nn_plates
from PIL import Image


IMG_WIDTH = conv_nn_plates.IMG_WIDTH
IMG_HEIGHT = conv_nn_plates.IMG_HEIGHT
PATH = 'E:/Study/Mallenom/test.jpg'


def normalize(image):
    return image[0:] / 255


def read(path):
    image = io.imread(path)
    return skimage.img_as_float(image)


def read_and_normalize(path):
    image = transform.resize(read(path), [IMG_HEIGHT, IMG_WIDTH, 3], mode='reflect')
    return image


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

#image = io.imread(PATH)
#img = transform.resize(image, [480, 640], mode='reflect')
#io.imsave(PATH, img)
