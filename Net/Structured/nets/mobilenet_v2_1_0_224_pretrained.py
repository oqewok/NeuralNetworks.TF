import tensorflow as tf
from skimage import io
from Structured.utils.img_preproc import *


import numpy as np
import os

np.set_printoptions(threshold=np.nan, suppress=True)

def get_mobilenet_pretrained():
    #with open("C:\\Users\\admin\\Documents\\GeneralProjectData\\mobilenet\\mobilenet_v2_1.0_224\\mobilenet_v2_1.0_224_frozen.pb", mode='rb') as f:
    with open("E:\\data\\mobilenet_v2_1.0_224\\mobilenet_v2_1.0_224_frozen.pb", mode='rb') as f:
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
        feature_map = graph.get_tensor_by_name("import/MobilenetV2/Conv_1/Relu6:0")
        feature_map = tf.reshape(feature_map, [1, 7, 7, 1280])
        #feature_map = graph.get_tensor_by_name("import/MobilenetV2/Conv_1/BatchNorm/FusedBatchNorm:0")

        return input, feature_map, [7, 7, 1280]


if __name__ == '__main__':
    get_mobilenet_pretrained()