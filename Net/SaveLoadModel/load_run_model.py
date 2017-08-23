import tensorflow as tf

with tf.Session() as sess:
    path = "F:\\Projects\\Python\\Tensorflow\\NeuralNetworks.TF\\Net\\SaveLoadModel\\models\\"
    loading_model = tf.train.import_meta_graph("%s%s" % (path, "my-model-step_final.meta"))
    loading_model.restore(sess, tf.train.latest_checkpoint(path))


    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")

    y_conv = graph.get_tensor_by_name("y_conv:0")

    print("end")