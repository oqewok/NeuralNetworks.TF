from Structured.utils.config import process_config
from Structured.utils.img_preproc import *

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import os

np.set_printoptions(threshold=np.nan, suppress=True)

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\experiments\\platenet_test_2\\"
)

#image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\Licence_plates_artificial\\img\\00000000_E112YB13.png")
image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\10 imgs\\file10038.jpg")

img = resize_img(image, [64, 128, 1], as_int=True)

saver = tf.train.import_meta_graph(os.path.join(path, "model.meta"))
graph = tf.get_default_graph()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver.restore(sess, os.path.join(path, "model"))

inputs = graph.get_tensor_by_name('Placeholder:0')
dropout = graph.get_tensor_by_name('Placeholder_2:0')
boxes = graph.get_tensor_by_name('add_5:0')

b = boxes.eval(session=sess, feed_dict={
    inputs: [img],
    dropout: 1.0,
})
b = (b + 1) * (64, 32, 64, 32)

plt.gray()

for i, box in enumerate(b):
    "show img"
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
        edgecolor='r', facecolor='none')
    gca = plt.gca()
    gca.add_patch(rect)
    plt.imshow(img)
    plt.show()
pass
