import tensorflow as tf
import numpy as np
import data_reader as reader
import image_proc
import train_localization as train

from skimage import novice
from skimage import io

PATH = 'E:/Study/Mallenom/hst.jpg'

np.set_printoptions(threshold=np.nan, suppress=True)

# print('Data batching...')
# batched_data = train_classification.get_batched_data(1)

picture = novice.open(PATH)
width = picture.width
height = picture.height

original_img = io.imread(PATH)
img = reader.read_image(PATH)
image = [img]


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('%s/model.meta' % train.MODEL_FOLDER)
    saver.restore(sess, tf.train.latest_checkpoint(train.MODEL_FOLDER))
    ops = tf.get_collection(
        'ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('inputs:0')
    y = graph.get_tensor_by_name('outputs:0')
    dropout = graph.get_tensor_by_name('fc2_c_dropout:0')
    # train_step = graph.get_operation_by_name('Adam')

    sess.run(tf.global_variables_initializer())

    f = open('E:/Study/Mallenom/img.txt', 'w')

    print('Evaluating...')

    w = y.eval(feed_dict={x: image, dropout: 1.0})[0]
    print(w)
    f.write(np.array2string(w, separator=','))
    f.close()

    image_proc.show_image(original_img, w, width, height)
