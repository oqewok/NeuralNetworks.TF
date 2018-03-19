import numpy as np
import tensorflow as tf

def generate_anchors(base_size, anchor_stride, aspect_ratios, scales, feature_map_shape):
    """Generate anchor for an image.
    Using the feature map, the output of the pretrained network for an
    image, and the anchor_reference generated using the anchor config
    values. We generate a list of anchors.
    Anchors are just fixed bounding boxes of different ratios and sizes
    that are uniformly generated throught the image.
    Args:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN. Should be (batch, height, width, depth).
    Returns:
        all_anchors: A flattened Tensor with all the anchors of shape
            `(num_anchors_per_points * feature_width * feature_height, 4)`
            using the (x1, y1, x2, y2) convention.
    """
    with tf.variable_scope('generate_anchors'):
        grid_width = feature_map_shape[2]  # width
        grid_height = feature_map_shape[1]  # height
        shift_x = tf.range(grid_width) * anchor_stride
        shift_y = tf.range(grid_height) * anchor_stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = tf.transpose(shifts)
        # Shifts now is a (H x W, 4) Tensor

        anchor_reference = generate_anchors_reference(
            base_size, aspect_ratios, scales
        )
        # Expand dims to use broadcasting sum.
        all_anchors = (
            np.expand_dims(anchor_reference, axis=0) +
            tf.expand_dims(shifts, axis=1)
        )

        # Flatten
        all_anchors = tf.reshape(
            all_anchors, (-1, 4)
        )
        return all_anchors


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.
    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.
    Scales apply to area of object.
    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.
    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )

    return anchors


"""
def generate_anchors(img_shape, conv_feats_shape, scales, ratios):
     Generate the anchors.
    print("Building the anchors...")
    ih, iw = img_shape
    fh, fw = conv_feats_shape
    n = fh * fw

    # Compute the coordinates of the anchors
    j = np.array(list(range(fh)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, fw))
    j = j.reshape((-1))

    i = np.array(list(range(fw)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (fh, 1))
    i = i.reshape((-1))

    s = np.ones((len(scales), n)) * scales.reshape(-1, 1)
    r = np.ones((len(ratios), n)) * ratios.reshape(-1, 1)

    # s = [5, 10, 15] => [[5, 10, 15], [5, 10, 15], ... (len(s) * len(r) times)]
    # r = [1, 2, 3]   => [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1], [2, 2, 2], [3, 3, 3]... len(r) times)]
    r = np.repeat(r, len(scales), axis=0)
    s = np.tile(s, (len(ratios), 1))

    # ratios:          1  1   1   2   2   2   3   3   3
    # w = s[i]*r[j] = [5, 10, 15, 10, 20, 30, 15, 30, 45]
    # h = s[i]      = [5, 10, 15, 5,  10, 15, 5,  10, 15]
    w = s * r
    h = s * 1
    # x = (i + 0.5) * iw / fw - w * 0.5
    # y = (j + 0.5) * ih / fh - h * 0.5
    x = i * iw / fw
    y = j * ih / fh

    # Determine if the anchors cross the boundary
    anchor_is_untruncated = np.ones((n), np.int32)
    anchor_is_untruncated[np.where(y < 0)[0]] = 0
    anchor_is_untruncated[np.where(x < 0)[0]] = 0
    anchor_is_untruncated[np.where(h + y > ih)[0]] = 0
    anchor_is_untruncated[np.where(w + x > iw)[0]] = 0

    # Clip the anchors if necessary
    x = np.maximum(x, w / 2)
    y = np.maximum(y, h / 2)
    w = np.minimum(w, (iw - x) * 2)
    h = np.minimum(h, (ih - y) * 2)

    x = np.expand_dims(x, 2)
    y = np.expand_dims(y, 2)
    w = np.expand_dims(w, 2)
    h = np.expand_dims(h, 2)
    anchors = np.concatenate((x, y, w, h), axis=2)

    anchors = np.unique(
        np.array(anchors, np.int32), axis=1)

    return anchors
"""

"""
anchors = generate_anchors_reference(128, [1, 4, 6], [0.5, 1, 2, 4])
pass
"""