import tensorflow as tf
from skimage import io
from Structured.utils.img_preproc import *

def get_vgg16_pretrained():
    with open("C:\\Users\\admin\Documents\\GeneralProjectData\\vgg\\vgg16.tfmodel", mode='rb') as f:
      fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={
        "images": images
    })
    print("graph loaded from disk")

    graph = tf.get_default_graph()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("variables initialized")

        ops = graph.get_operations()
        feature_map = graph.get_tensor_by_name("import/conv5_3/Relu:0")

        feats_shape = list(np.array(feature_map.shape[1:4], np.int32))
        #prob_tensor = graph.get_tensor_by_name("import/prob:0")

        return images, feature_map, feats_shape
