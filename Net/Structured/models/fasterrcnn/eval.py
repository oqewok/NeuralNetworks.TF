from Structured.utils.config import process_config
from Structured.utils.img_preproc import *

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import os

np.set_printoptions(threshold=np.nan, suppress=True)

config = process_config("C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\fastercnn.json")

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\faster-rcnn\\checkpoint\\faster-rcnn.meta"
)

image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\images\\Казахстан(KZ)\\full_images_KZ\\image33764.jpg")
img = resize_img(image, config.input_shape, as_int=True)


with tf.Session() as sess:
    saver       = tf.train.import_meta_graph(path)
    saver.restore(
        sess, tf.train.latest_checkpoint(
            "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\faster-rcnn\\checkpoint"
        )
    )

    ops         = tf.get_collection('ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

    graph       = tf.get_default_graph()
    inputs      = graph.get_tensor_by_name('inputs:0')
    bboxes      = graph.get_tensor_by_name('BoundingBoxTransform/clip_bboxes_1/concat:0')
    scores      = graph.get_tensor_by_name('nms/gather_nms_proposals_scores:0')
    is_train    = graph.get_tensor_by_name('is_train:0')

    sess.run(tf.global_variables_initializer())

    result = bboxes.eval(
        feed_dict={
            inputs: [img],
            is_train: False
        }
    )

    "show img"
    fig, ax = plt.subplots(1)

    ax.imshow(img)

    for i in range(5):
        rect = patches.Rectangle(
            (result[i][0], result[i][1]), result[i][2], result[i][3], linewidth=1,
            edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
pass
