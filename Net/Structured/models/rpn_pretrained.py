import tensorflow as tf
from skimage import io
from Structured.utils.img_preproc import *
import os

def get_rpn_pretrained():
    path = os.path.join(
        r"C:\Users\admin\Documents\GeneralProjectData\Projects\NeuralNetworks.TF\Net\Structured\experiments\mobilenet_v1_1_0_rpn_10650_1\checkpoint\mobilenet_v1_1_0_rpn_10650_1.meta"
    )

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.import_meta_graph(path)
        saver.restore(
            sess, tf.train.latest_checkpoint(
                r"C:\Users\admin\Documents\GeneralProjectData\Projects\NeuralNetworks.TF\Net\Structured\experiments\mobilenet_v1_1_0_rpn_10650_1\checkpoint"
            )
        )

        graph = tf.get_default_graph()

        images = graph.get_tensor_by_name('inputs:0')
        is_train = graph.get_tensor_by_name('is_train:0')
        feature_map = graph.get_tensor_by_name("import/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:0")
        rpn_roi_proposals = graph.get_tensor_by_name('BoundingBoxTransform/clip_bboxes_1/concat:0')
        rpn_cls_scores = graph.get_tensor_by_name('nms/gather_nms_proposals_scores:0')
        gt_boxes = graph.get_tensor_by_name('gt_boxes:0')
        rpn_cls_loss = graph.get_tensor_by_name('RPNLoss/Sum_1:0')
        rpn_reg_loss = graph.get_tensor_by_name('RPNLoss/Sum_2:0')

        return images, is_train, feature_map, rpn_roi_proposals, rpn_cls_scores, gt_boxes, rpn_cls_loss, rpn_reg_loss
