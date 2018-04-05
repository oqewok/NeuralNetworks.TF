from Structured.utils.config import process_config
from Structured.utils.img_preproc import *
from Structured.data_loader.parser import COUNTRIES

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import os

np.set_printoptions(threshold=np.nan, suppress=True)

config = process_config(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\platenet.json")

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\platenet_8\\checkpoint\\"
)

image = io.imread("C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\plates\\foto14992.jpg")
#image = subtract_channels(image)
img = resize_img(image, config.input_shape, as_int=True)
img /= 255

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join(path, "platenet_8.meta"))
saver.restore(
    sess, os.path.join(path, "platenet_8")
    )

graph = tf.get_default_graph()
inputs = graph.get_tensor_by_name('inputs:0')
boxes = graph.get_tensor_by_name('bboxes/xw_plus_b:0')
probs = graph.get_tensor_by_name('Softmax:0')
is_train = graph.get_tensor_by_name('is_train:0')
# conv1 = graph.get_tensor_by_name('Relu_2:0')
#
# c = conv1.eval(session=sess, feed_dict={
#     inputs: [img],
#     is_train: False,
# })[0][:, :, 99]

b, p = sess.run([boxes, probs], feed_dict={
    inputs: [img],
    is_train: False,
})

H, W, C = config.input_shape

b = (b + 1.0) * (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H)
p = np.argmax(p)

"show img"
fig, ax = plt.subplots(1)

xmin, ymin, xmax, ymax = b[0]

b = adjust_bboxes(np.array([[xmin, ymin, xmax, ymax, p]]), img.shape, image.shape)[0]
rect = patches.Rectangle(
    (b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1,
    edgecolor='r', facecolor='none')
ax.add_patch(rect)

for v, k in COUNTRIES.items():
    if k == p:
        print(v)

ax.imshow(image)
plt.show()


pass
