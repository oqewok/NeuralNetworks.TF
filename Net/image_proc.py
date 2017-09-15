import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import numpy as np
from skimage import io
from skimage import transform
from Net import conv_nn_plates_light
from PIL import Image

IMG_WIDTH = conv_nn_plates_light.IMG_WIDTH
IMG_HEIGHT = conv_nn_plates_light.IMG_HEIGHT
PATH = 'E:/Study/Mallenom/test.jpg'


def normalize(image):
    return image[0:] / 255


def read(path):
    image = io.imread(path)
    return skimage.img_as_float(image)


def read_and_normalize(path):
    image = transform.resize(read(path), [IMG_HEIGHT, IMG_WIDTH, conv_nn_plates_light.CHANNELS], mode='reflect')
    return image


def show_image(image, coords, width, height, original=None):
    fig, ax = plt.subplots(1)

    ax.imshow(image)

    coords = decode_rect(coords, width, height)
    rect = patches.Rectangle(
        (coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1], linewidth=1,
        edgecolor='r', facecolor='none')
    ax.add_patch(rect)


    if original != None:
        original = decode_rect(original, width, height)
        r = patches.Rectangle(
            (original[0], original[1]), original[2] - original[0], original[3] - original[1],
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(r)

    plt.show()


def decode_rect(coords, width, height):
    result = np.copy(coords)
    for i in range(0, len(coords)):
        if i % 2 == 0:
            result[i] = int(width * coords[i])
        else:
            result[i] = int(height * coords[i])

    return result

# image = read(PATH)
# img = skimage.img_as_float(image)
# im = np.array(img)
# a = np.reshape(img, 480 * 640 * 3)
# img = Image.open(PATH)
# im = img.resize((96, 128))
# i = np.array(im) / 255
# print()
# n_image = normalize(image)
# show_image(n_image)

# image = io.imread(PATH)
# img = transform.resize(image, [480, 640], mode='reflect')
# io.imsave(PATH, img)
