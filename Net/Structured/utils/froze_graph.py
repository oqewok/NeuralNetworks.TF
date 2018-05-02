import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os

path = os.path.join(
    "E:\\data\\checkpoint\\vgg16_rpn_10500_2.meta"
)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path)
    saver.restore(
        sess, "E:\\data\\checkpoint\\vgg16_rpn_10500_2"
        )

    dir = "E:\\data\\checkpoint\\"

    graph_def = sess.graph_def

    frozen_graph_def = convert_variables_to_constants(sess, graph_def, ["BoundingBoxTransform/clip_bboxes_1/concat", "nms/gather_nms_proposals_scores"])

    pb_frozen_file = os.path.join(
        dir,
        "fasterrcnn.pb")

    with tf.gfile.GFile(pb_frozen_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
