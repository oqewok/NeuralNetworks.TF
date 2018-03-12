import tensorflow as tf
import numpy as np


class RPN:
    """RPN - Region Proposal Network.
            Given an image (as feature map) and a fixed set of anchors, the RPN
            will learn weights to adjust those anchors so they better look like the
            ground truth objects, as well as scoring them by "objectness" (ie. how
            likely they are to be an object vs background).
            The final result will be a set of rectangular boxes ("proposals"),
            each associated with an objectness score.
            Note: this module can be used independently of Faster R-CNN.

            Дано изображение (представленное в виде feature maps) и фиксированных набор "якорей", RPN
            будет обучать веса для регулировки этих "якорей", чтобы они лучше соответствовали
            ground truth объектам, также оценивая их результат на "предметность (objectness)"
            (к чему их относить - к объекту или фону).
            Конечным результатом является набор прямоугольных областей ("предположений (proposals))",
            каждая из которых ассоциирована со своим результатом на "предметность (objectness)".

            Замечание: данный модуль может быть использован независимо от Faster R-CNN.
    """
    def __init__(self, config):
        self.config = config

        self.anchor_sizes  = np.array(config.anchor_sizes)
        self.anchor_ratios = np.array(config.anchor_ratios)
        self.anchors_count = len(self.anchor_sizes) * len(self.anchor_ratios)



    def build(self, img_shape, conv_feats, conv_feats_shape):

        print("Building the anchors...")
        self.img_shape = np.array(img_shape[:2], np.int32)

        self.conv_feats_shape = np.array(conv_feats_shape[:2], np.int32)


    def generate_anchors(self, scale, ratio, factor=1.5):
        """ Generate the anchors. """
        ih, iw = self.img_shape
        fh, fw = self.conv_feats_shape
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

        s = np.ones((n)) * scale
        r0 = np.ones((n)) * ratio[0]
        r1 = np.ones((n)) * ratio[1]

        h = s * r0
        w = s * r1
        y = (j + 0.5) * ih / fh - h * 0.5
        x = (i + 0.5) * iw / fw - w * 0.5

        ph = h * factor
        pw = w * factor
        py = y - h * (factor * 0.5 - 0.5)
        px = x - w * (factor * 0.5 - 0.5)

        # Determine if the anchors cross the boundary
        anchor_is_untruncated = np.ones((n), np.int32)
        anchor_is_untruncated[np.where(y < 0)[0]] = 0
        anchor_is_untruncated[np.where(x < 0)[0]] = 0
        anchor_is_untruncated[np.where(h + y > ih)[0]] = 0
        anchor_is_untruncated[np.where(w + x > iw)[0]] = 0

        parent_anchor_is_untruncated = np.ones((n), np.int32)
        parent_anchor_is_untruncated[np.where(py < 0)[0]] = 0
        parent_anchor_is_untruncated[np.where(px < 0)[0]] = 0
        parent_anchor_is_untruncated[np.where(ph + py > ih)[0]] = 0
        parent_anchor_is_untruncated[np.where(pw + px > iw)[0]] = 0

        # Clip the anchors if necessary
        y = np.maximum(y, np.zeros((n)))
        x = np.maximum(x, np.zeros((n)))
        h = np.minimum(h, ih - y)
        w = np.minimum(w, iw - x)

        py = np.maximum(py, np.zeros((n)))
        px = np.maximum(px, np.zeros((n)))
        ph = np.minimum(ph, ih - py)
        pw = np.minimum(pw, iw - px)

        y = np.expand_dims(y, 1)
        x = np.expand_dims(x, 1)
        h = np.expand_dims(h, 1)
        w = np.expand_dims(w, 1)
        anchors = np.concatenate((y, x, h, w), axis=1)
        anchors = np.array(anchors, np.int32)

        py = np.expand_dims(py, 1)
        px = np.expand_dims(px, 1)
        ph = np.expand_dims(ph, 1)
        pw = np.expand_dims(pw, 1)
        parent_anchors = np.concatenate((py, px, ph, pw), axis=1)
        parent_anchors = np.array(parent_anchors, np.int32)

        # Count the number of untruncated anchors
        num_anchor = np.array([n], np.int32)
        num_untruncated_anchor = np.sum(anchor_is_untruncated)
        num_untruncated_anchor = np.array([num_untruncated_anchor], np.int32)
        num_untruncated_parent_anchor = np.sum(parent_anchor_is_untruncated)
        num_untruncated_parent_anchor = np.array([num_untruncated_parent_anchor], np.int32)

        return num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor

""" Generate the anchors. """
# anchor_sizes = [16, 32, 64] => factor = 2.
factor = 2
scale = 16
ratio = [4, 1]

ih, iw = 600, 800
fh, fw = 39, 51
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

s = np.ones((n)) * scale
r0 = np.ones((n)) * ratio[0]
r1 = np.ones((n)) * ratio[1]

h = s * r0
w = s * r1
y = (j + 0.5) * ih / fh - h * 0.5
x = (i + 0.5) * iw / fw - w * 0.5

ph = h * factor
pw = w * factor
py = y - h * (factor * 0.5 - 0.5)
px = x - w * (factor * 0.5 - 0.5)

# Determine if the anchors cross the boundary
anchor_is_untruncated = np.ones((n), np.int32)
anchor_is_untruncated[np.where(y < 0)[0]] = 0
anchor_is_untruncated[np.where(x < 0)[0]] = 0
anchor_is_untruncated[np.where(h + y > ih)[0]] = 0
anchor_is_untruncated[np.where(w + x > iw)[0]] = 0

parent_anchor_is_untruncated = np.ones((n), np.int32)
parent_anchor_is_untruncated[np.where(py < 0)[0]] = 0
parent_anchor_is_untruncated[np.where(px < 0)[0]] = 0
parent_anchor_is_untruncated[np.where(ph + py > ih)[0]] = 0
parent_anchor_is_untruncated[np.where(pw + px > iw)[0]] = 0

# Clip the anchors if necessary
y = np.maximum(y, np.zeros((n)))
x = np.maximum(x, np.zeros((n)))
h = np.minimum(h, ih - y)
w = np.minimum(w, iw - x)

py = np.maximum(py, np.zeros((n)))
px = np.maximum(px, np.zeros((n)))
ph = np.minimum(ph, ih - py)
pw = np.minimum(pw, iw - px)

y = np.expand_dims(y, 1)
x = np.expand_dims(x, 1)
h = np.expand_dims(h, 1)
w = np.expand_dims(w, 1)
# TODO: Task7: М.б. лучше concatenate((x, y, h, w) ???
anchors = np.concatenate((y, x, h, w), axis=1)
anchors = np.array(anchors, np.int32)

py = np.expand_dims(py, 1)
px = np.expand_dims(px, 1)
ph = np.expand_dims(ph, 1)
pw = np.expand_dims(pw, 1)
parent_anchors = np.concatenate((py, px, ph, pw), axis=1)
parent_anchors = np.array(parent_anchors, np.int32)

# Count the number of untruncated anchors
num_anchor = np.array([n], np.int32)
num_untruncated_anchor = np.sum(anchor_is_untruncated)
num_untruncated_anchor = np.array([num_untruncated_anchor], np.int32)
num_untruncated_parent_anchor = np.sum(parent_anchor_is_untruncated)
num_untruncated_parent_anchor = np.array([num_untruncated_parent_anchor], np.int32)

pass