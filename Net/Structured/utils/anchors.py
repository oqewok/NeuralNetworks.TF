import numpy as np


def generate_anchors(img_shape, conv_feats_shape, scales, ratios):
    """ Generate the anchors. """
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