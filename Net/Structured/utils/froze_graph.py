import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os

path = os.path.join(
    "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\vgg16_rpn_1000_7\\checkpoint\\vgg16_rpn_1000_7.meta"
)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(path)
    saver.restore(
        sess, tf.train.latest_checkpoint(
            "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\vgg16_rpn_1000_7\\checkpoint"
        )
    )

    dir = "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\experiments\\vgg16_rpn_1000_7\\"

    # pb_file = os.path.join(
    #     dir,
    #     "fasterrcnn.pb")
    #
    # with open(pb_file, 'wb') as f:
    #     f.write(sess.graph_def.SerializeToString())
    #
    # with gfile.FastGFile(pb_file, 'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())

    frozen_graph_def = convert_variables_to_constants(sess, sess.graph_def, ["BoundingBoxTransform/clip_bboxes_1/concat", "nms/gather_nms_proposals_scores"])

    pb_frozen_file = os.path.join(
        dir,
        "fasterrcnn_frozen.pb")

    with tf.gfile.GFile(pb_frozen_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
