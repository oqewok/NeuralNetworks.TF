import tensorflow as tf

# Определение переменных
w1 = tf.Variable(5, dtype=tf.float32, name="w1")
w2 = tf.Variable(3, dtype=tf.float32, name="w2")

# Определение набора данных
# (Позволяет подставлять нужное значение по именам переменных без замены значения самих переменных)
new_data = {w1: 12.0, w2: 8.0}

# Определение операции (умножить w1 на w2)
result = tf.multiply(w1, w2, name="result")

var_initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(var_initializer)

    print(sess.run(result))                 # 15
    print(sess.run(result, new_data))       # 96 – т.к. w1, w2 вызваны со значениями из new_data
    print(sess.run(result))                 # 15

    saver.save(sess, './models/multiply/final_model')
    print("saved")

print("end")
