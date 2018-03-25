import skimage
import tensorflow as tf
import numpy as np

from skimage import io
from skimage import transform

# Default RGB means used commonly.
_R_MEAN = 98.1689309323291
_G_MEAN = 99.10622145031063
_B_MEAN = 97.50672213484069

# _R_MEAN = 123.68
# _G_MEAN = 116.78
# _B_MEAN = 103.94


# Изменение размера изображения
def resize_img(image, new_shape, bboxes=None, as_int=True):
    old_shape = image.shape

    try:
        image = transform.resize(
            image, new_shape, mode='reflect', preserve_range=as_int)

        if as_int:
            image = np.array(image, int)

        if bboxes is not None:
            bboxes = adjust_bboxes(bboxes, old_shape, new_shape)
            return image, bboxes

        return image
    except IndexError:
        print("it's a trap!")


def adjust_bboxes(bboxes, old_shape, new_shape):
    """Adjusts the bboxes of an image that has been resized.
    Args:
        bboxes: Tensor with shape (num_bboxes, 5). Last element is the label.
        old_height: Float. Height of the original image.
        old_width: Float. Width of the original image.
        new_height: Float. Height of the image after resizing.
        new_width: Float. Width of the image after resizing.
    Returns:
        Tensor with shape (num_bboxes, 5), with the adjusted bboxes.
    """

    old_height, old_width = old_shape[0], old_shape[1]
    new_height, new_width = new_shape[0], new_shape[1]

    # We normalize bounding boxes points.
    bboxes_float = np.array(bboxes, dtype=np.float32)
    x_min, y_min, x_max, y_max, label = np.split(bboxes_float, 5, axis=1)

    x_min = x_min / old_width
    y_min = y_min / old_height
    x_max = x_max / old_width
    y_max = y_max / old_height

    # Use new size to scale back the bboxes points to absolute values.
    x_min = np.int32(x_min * new_width)
    y_min = np.int32(y_min * new_height)
    x_max = np.int32(x_max * new_width)
    y_max = np.int32(y_max * new_height)
    label = np.int32(label)  # Cast back to int.

    # Concat points and label to return a [num_bboxes, 5] tensor.
    return np.concatenate((x_min, y_min, x_max, y_max, label), axis=1)


def preprocess(inputs):
    inputs = subtract_channels(inputs)
    inputs = normalize(inputs)

    return inputs


def subtract_channels(inputs, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtract channels from images.
    It is common for CNNs to subtract the mean of all images from each
    channel. In the case of RGB images we first calculate the mean from
    each of the channels (Red, Green, Blue) and subtract those values
    for training and for inference.
    Args:
        inputs: A Tensor of images we want to normalize. Its shape is
            (1, height, width, num_channels).
        means: A Tensor of shape (num_channels,) with the means to be
            subtracted from each channels on the inputs.
    Returns:
        outputs: A Tensor of images normalized with the means.
            Its shape is the same as the input.
    """
    return inputs - [means]


def normalize(inputs):
    """Normalize between -1.0 to 1.0.
    Args:
        inputs: A Tensor of images we want to normalize. Its shape is
            (1, height, width, num_channels).
    Returns:
        outputs: A Tensor of images normalized between -1 and 1.
            Its shape is the same as the input.
    """
    inputs = inputs / 255.
    inputs = (inputs - 0.5) * 2.
    return inputs


'''
def resize_batch(imgs, new_shape):
    for i in range(len(imgs)):
        imgs[i] = resize_img(imgs[i], new_shape)

    return imgs


# Преобразование значений пикселей в интервал [0.0, 1.0]
def as_float(image):
    image = skimage.img_as_float(image)
    return image
'''

'''
bbox = [[10, 10, 30, 30, 1], [22, 18, 46, 38, 1]]
bbox_new = adjust_bboxes(bbox, [100, 100], [50, 50])

print(bbox_new)
'''