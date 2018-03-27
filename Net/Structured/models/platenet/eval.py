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
img /= 255.0

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
    probs = graph.get_tensor_by_name('Softmax:0')

    is_train = graph.get_tensor_by_name('is_train:0')

    sess.run(tf.global_variables_initializer())

    prob = sess.run(
        [probs], feed_dict={inputs: [img], is_train: False}
    )

    print(prob)

    "show img"
    fig, ax = plt.subplots(1)

    ax.imshow(img)
    plt.show()
pass
