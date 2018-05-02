import tensorflow as tf
from skimage import io
from Structured.utils.img_preproc import *


import numpy as np
import os

np.set_printoptions(threshold=np.nan, suppress=True)

def get_mobilenet_v1_0_75_pretrained():
    with open(
            "C:\\Users\\admin\\Documents\\GeneralProjectData\\mobilenet\\mobilenet_v1_0.75_224\\mobilenet_v1_0.75_224_frozen.pb",
            mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    input = tf.placeholder("float", [None, 224, 224, 3], name="inputs")

    tf.import_graph_def(graph_def, input_map={
        "input": input
    })

    print("graph loaded from disk")

    graph = tf.get_default_graph()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        #ops = graph.get_operations()
        #print(list(ops))

        #feature_map = graph.get_tensor_by_name("import/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6:0")
        feature_map = graph.get_tensor_by_name("import/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6:0")
        feats_shape = list(np.array(feature_map.shape[1:4], np.int32))

        return input, feature_map, feats_shape


if __name__ == '__main__':
    get_mobilenet_v1_0_75_pretrained()
