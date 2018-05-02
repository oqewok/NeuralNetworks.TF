import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os

path = os.path.join(
    "E:\\data\\checkpoint\\vgg16_rpn_10500_2.meta"
)

from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim

inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="inputs")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net, end_points = resnet_v2.resnet_v2_50(inputs, is_training=False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(
        sess, "C:\\Users\\admin\\Documents\\GeneralProjectData\\resnet\\resnet_v2_50\\resnet_v2_50.ckpt"
        )

    dir = "C:\\Users\\admin\\Documents\\GeneralProjectData\\resnet\\resnet_v2_50\\"

    a = sess.graph.get_tensor_by_name('resnet_v2_50/block3/unit_5/bottleneck_v2/add:0')

    graph_def = sess.graph_def

    frozen_graph_def = convert_variables_to_constants(sess, graph_def, ['resnet_v2_50/block3/unit_5/bottleneck_v2/add:0'])

    pb_frozen_file = os.path.join(
        dir,
        "resnet_v2_50.pb")

    with tf.gfile.GFile(pb_frozen_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
