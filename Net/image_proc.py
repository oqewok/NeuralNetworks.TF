import matplotlib.pyplot as plt
import skimage
from skimage import io

IMG_WIDTH = 128
IMG_HEIGHT = 96
PATH = 'E:/data/gt_db/s01/01.jpg'


def normalize(image):
    return image[0:]/255


def read(path):
    image = io.imread(path)
    return image

def read_and_normalize(path):
    image = read(path)
    return normalize(image)

def show_image(image):
    plt.imshow(image)
    plt.waitforbuttonpress()


# image = read(PATH)
# n_image = normalize(image)
# show_image(n_image)


