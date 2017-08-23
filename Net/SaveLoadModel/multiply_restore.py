# Восстановление и запуск модели
# Необходимо указать путь до моделей (path)

import tensorflow as tf

with tf.Session() as sess:


    path = "F:\\Projects\\Python\\Tensorflow\\NeuralNetworks.TF\\Net\\SaveLoadModel\\models\\multiply\\"

    # Восстановление графа
    loading_model = tf.train.import_meta_graph("%s%s" % (path, "final_model.meta"))

    # Восстановление переменных в текущую сессию
    loading_model.restore(sess, tf.train.latest_checkpoint(path))

    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())

    # Восстановление переменных по имени
    # ("связывание" переменных модели с переменными проограммы)
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")

    # Восстановление операции по имени
    # result = tf.multiply(w1, w2, name="result")
    result = graph.get_tensor_by_name("result:0")

    # Новый набор данных
    new_data = {w1: 10.0, w2: 15}

    # Запуск операции графа загруженной модели с новыми данными
    print(sess.run(result, new_data))

    print("end")