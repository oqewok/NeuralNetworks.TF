import skimage
import tensorflow as tf
import numpy as np

from skimage import io
from skimage import transform


# Изменение размера изображения
def resize_img(image, new_shape, bboxes=None, as_int=True):
    old_shape = image.shape

    image = transform.resize(
            image, new_shape, mode='reflect', preserve_range=as_int)

    if as_int:
        image = np.array(image, int)

    if bboxes is not None:
        bboxes = adjust_bboxes(bboxes, old_shape, new_shape)
        return image, bboxes

    return image


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

    bboxes_float = np.array(bboxes, dtype=float)
    result_boxes = []

    for bbox in bboxes_float:
        #x_min, y_min, x_max, y_max, label = tf.unstack(bboxes_float, axis=1)
        x_min, y_min, x_max, y_max = bbox

        x_min = x_min / old_width
        y_min = y_min / old_height
        x_max = x_max / old_width
        y_max = y_max / old_height

        # Use new size to scale back the bboxes points to absolute values.
        x_min = x_min * new_width
        y_min = y_min * new_height
        x_max = x_max * new_width
        y_max = y_max * new_height

        b = np.stack((x_min, y_min, x_max, y_max))
        result_boxes.append(np.array(b, dtype=int))

    # Concat points and label to return a [num_bboxes, 5] tensor.

    return np.array(result_boxes)

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
bbox = [[10, 10, 30, 30], [22, 18, 46, 38]]
bbox_new = adjust_bboxes(bbox, [100, 100], [60, 40])
print()
'''
