from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim

from Structured.utils.img_preproc import *
from Structured.utils.utils import get_args
from Structured.utils.config import process_config

def get_resnet_v2_pretrained():
    # Create graph
    args = get_args()
    config = process_config(args.config)

    H, W, C = config.input_shape

    inputs = tf.placeholder(tf.float32, shape=[None, H, W, C], name="inputs")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_50(inputs, is_training=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'E:\\data\\resnet_v2_50\\resnet_v2_50.ckpt')

        feature_map = sess.graph.get_tensor_by_name('resnet_v2_50/block3/unit_5/bottleneck_v2/add:0')
        feats_shape = list(np.array(feature_map.shape[1:4], np.int32))

        return inputs, feature_map, feats_shape


if __name__ == '__main__':
    get_resnet_v2_pretrained()