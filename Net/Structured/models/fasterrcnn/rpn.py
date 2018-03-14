import tensorflow as tf
import numpy as np

from Structured.utils.anchors import generate_anchors

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
    def __init__(self, config, conv_feats, conv_feats_shape):
        self.config           = config

        self.conv_feats       = conv_feats
        self.conv_feats_shape = conv_feats_shape

        self.anchor_scales    = np.array(config.anchor_sizes)
        self.anchor_ratios    = np.array(config.anchor_ratios)
        self.anchors_count    = len(self.anchor_scales) * len(self.anchor_ratios)

        self.build()

    def build(self):

        self.img_shape          = np.array(self.img_shape[:2], np.int32)
        self.conv_feats_shape   = np.array(self.conv_feats_shape[:2], np.int32)

        """ Generate the anchors. """
        self.anchors            = generate_anchors(
            self.img_shape, self.conv_feats_shape, self.anchor_scales, self.anchor_ratios)


""" Generate the anchors.
# anchor_scales = [32, 64, 128] => factor = 2.
scales = np.array([32, 64, 128], dtype=int)
ratios = np.array([1, 4, 6], dtype=int)

ih, iw = 600, 800
fh, fw = 39, 51
n      = fh * fw
print("Building the anchors...")

pass
"""