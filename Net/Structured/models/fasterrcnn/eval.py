from Structured.utils.config import process_config
from Structured.utils.img_preproc import *

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import os

np.set_printoptions(threshold=np.nan, suppress=True)

config = process_config(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\fastercnn.json")

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\rpn_3\\checkpoint\\rpn_3.meta"
)

image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\images\\file10038.jpg")
img = resize_img(image, config.input_shape, as_int=True)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path)
    saver.restore(
        sess, tf.train.latest_checkpoint(
            "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\rpn_3\\checkpoint"
        )
    )

    ops = tf.get_collection(
        'ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs:0')
    # rcnn
    if config.with_rcnn:
        bboxes = graph.get_tensor_by_name('rcnn_prediction/bboxes:0')
        labels = graph.get_tensor_by_name('rcnn_prediction/labels:0')
        probs = graph.get_tensor_by_name('rcnn_prediction/TopKV2:0')
    else:
        bboxes = graph.get_tensor_by_name('BoundingBoxTransform/clip_bboxes_1/concat:0')
        probs = graph.get_tensor_by_name('nms/gather_nms_proposals_scores:0')
        labels = None

    is_train = graph.get_tensor_by_name('is_train:0')

    sess.run(tf.global_variables_initializer())

    if config.with_rcnn:
        box, lab, prob = sess.run(
            [bboxes, labels, probs],
            feed_dict={
                inputs: [img],
                is_train: False
            }
        )

        filter = lab == 1
        box = box[filter]
    else:
        box, prob = sess.run(
            [bboxes, probs],
            feed_dict={
                inputs: [img],
                is_train: False
            }
        )


    for j in range(0, 10):
        "show img"
        fig, ax = plt.subplots(1)
        for i in range(0, 5):
            rect = patches.Rectangle(
                (box[j*i + i][0] - 0.5 * box[j*i + i][2], box[j*i + i][1] - 0.5 * box[j*i + i][3]), box[j*i + i][2], box[j*i + i][3], linewidth=1,
                edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(img)
        plt.show()
pass
