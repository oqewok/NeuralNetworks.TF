from Structured.utils.config import process_config
from Structured.utils.img_preproc import *

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import os

np.set_printoptions(threshold=np.nan, suppress=True)

config = process_config(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\platenet.json")

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\platenet_1\\checkpoint\\platenet_1.meta"
)

image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\Licence_plates_artificial\\00000002_C725PE08.png")
img = resize_img(image, config.input_shape, as_int=True)
img /= 255

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path)
    saver.restore(
        sess, tf.train.latest_checkpoint(
            "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\platenet_1\\checkpoint"
        )
    )

    ops = tf.get_collection(
        'ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs:0')
    boxes = graph.get_tensor_by_name('add_5:0')

    sess.run(tf.global_variables_initializer())

    b = boxes.eval(session=sess, feed_dict={inputs: [img]})[0]

    H, W, C = config.input_shape

    b = (b + 1.0) * (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H)
    "show img"
    fig, ax = plt.subplots(1)

    rect = patches.Rectangle(
        (b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1,
        edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    img = np.reshape(img, [img.shape[0], img.shape[1]])
    ax.imshow(image)
    plt.show()
pass
