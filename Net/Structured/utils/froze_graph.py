import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os

path = os.path.join(
    r"C:\Users\admin\Documents\GeneralProjectData\Projects\NeuralNetworks.TF\Net\Structured\experiments\mobilenet_v1_0_75_rpn_10650_1\checkpoint\mobilenet_v1_0_75_rpn_10650_1.meta"
)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path)
    saver.restore(
        sess, r"C:\Users\admin\Documents\GeneralProjectData\Projects\NeuralNetworks.TF\Net\Structured\experiments\mobilenet_v1_0_75_rpn_10650_1"
        )

    dir = r"C:\Users\admin\Documents\GeneralProjectData\Projects\NeuralNetworks.TF\Net\Structured\experiments\mobilenet_v1_0_75_rpn_10650_1"

    graph_def = sess.graph_def

    frozen_graph_def = convert_variables_to_constants(sess, graph_def, ["BoundingBoxTransform/clip_bboxes_1/concat", "nms/gather_nms_proposals_scores"])

    pb_frozen_file = os.path.join(
        dir,
        "fasterrcnn.pb")

    with tf.gfile.GFile(pb_frozen_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
