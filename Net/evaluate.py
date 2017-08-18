import tensorflow as tf
import numpy as np
import train

np.set_printoptions(threshold=np.nan, suppress=True)

print('Data batching...')
batched_data = train.get_batched_data(train.BATCH_SIZE)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    ops = tf.get_collection(
        'ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('inputs:0')
    y = graph.get_tensor_by_name('y_sigmoid:0')
    keep_prob = graph.get_tensor_by_name('Placeholder:0')
    train_step = graph.get_operation_by_name('Adam')

    sess.run(tf.global_variables_initializer())

    batch = train.next_batch(batched_data, 0)

    f = open('E:/Study/Mallenom/2.txt', 'w')
    w = y.eval(feed_dict={x: batch[0], keep_prob: 1.0})
    f.write(np.array2string(w, separator=','))
    f.close()
    print()

