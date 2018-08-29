import tensorflow as tf
from skimage import io
from Structured.utils.img_preproc import *


import numpy as np
import os

np.set_printoptions(threshold=np.nan, suppress=True)

def get_mobilenet_v2_1_0_feature_extractor():
    with open("C:\\Users\\admin\\Documents\\GeneralProjectData\\mobilenet\\mobilenet_v2_1.0_224\\mobilenet_v2_1.0_224_frozen.pb", mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    input = tf.placeholder("float", [None, 224, 224, 3], name="inputs")

    tf.import_graph_def(graph_def, input_map={
        "input": input
    })
    print("graph loaded from disk")

    graph = tf.get_default_graph()

    feature_maps = []

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = graph.get_operations()
        print(ops)

        feature_map1 = graph.get_tensor_by_name("import/MobilenetV2/expanded_conv_1/project/BatchNorm/FusedBatchNorm:0")
        feature_map1 = tf.reshape(feature_map1, [None, 56, 56, 24])
        tf.nn.l2_normalize(feature_map1, axis=3)
        feature_maps.append(feature_map1)

        feature_map2 = graph.get_tensor_by_name("import/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm:0")
        feature_map2 = tf.reshape(feature_map2, [None, 28, 28, 32])
        feature_maps.append(feature_map2)

        feature_map3 = graph.get_tensor_by_name("import/MobilenetV2/expanded_conv_12/project/BatchNorm/FusedBatchNorm:0")
        feature_map3 = tf.reshape(feature_map3, [None, 14, 14, 96])
        feature_maps.append(feature_map3)

        feature_map4 = graph.get_tensor_by_name("import/MobilenetV2/Conv_1/Relu6:0")
        feature_map4 = tf.reshape(feature_map4, [None, 7, 7, 1024])
        feature_maps.append(feature_map4)

        feature_map5 = graph.get_tensor_by_name("import/MobilenetV2/Logits/AvgPool:0")
        feature_map5 = tf.reshape(feature_map5, [None, 1, 1, 1024])
        feature_maps.append(feature_map5)

        #feature_map = graph.get_tensor_by_name("import/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm:0")
        # feature_map = graph.get_tensor_by_name("import/MobilenetV2/Conv_1/Relu6:0")
        # feature_map = graph.get_tensor_by_name("import/MobilenetV2/Conv_1/BatchNorm/FusedBatchNorm:0")
        #feature_map = tf.reshape(feature_map, [1, 7, 7, 1280])

        return input, feature_maps


if __name__ == '__main__':
    get_mobilenet_v2_1_0_feature_extractor()