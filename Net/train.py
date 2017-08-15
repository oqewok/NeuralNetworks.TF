import tensorflow as tf
import data_reader
import conv_nn_faces_model as model
import numpy as np

NUM_OF_EPOCHS = 10000
BATCH_SIZE = 10
LEARNING_RATE = 1e-4


def load_train_data():
    # здесь берем порциями данные
    names, labels = data_reader.read_labeled_image_list()
    data = data_reader.read_images_from_disk(names, labels)
    return data


def get_batched_data(data, batch_size):
    img, masks = data[0], data[1]
    batched_img = [img[i:i + batch_size] for i in range(0, 750, batch_size)]
    batched_masks = [masks[i:i + batch_size] for i in range(0, 750, batch_size)]
    return [batched_img, batched_masks]


def next_batch(batched_data, batch_index):
    i, l = batched_data[0], batched_data[1]
    images = i[batch_index]
    masks = l[batch_index]
    return [images, masks]


def get_loss(y, y_):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    return loss


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, y, params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, model.OutputNodesCount], name='outputs')

    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    print('Loading data...')
    data = load_train_data()
    print('Data batching...')
    batched_data = get_batched_data(data, batch_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        for step in range(num_of_epochs + 1):
            batch = next_batch(batched_data, step % batch_size)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if step % 1 == 0:
                print('Step', step, 'of', num_of_epochs)

                # print('Saving...')
                # saver.save(sess, './my-model')
                # print('Saving complete.')

        f = open('E:/Study/Mallenom/1.txt', 'w')
        w = y.eval(feed_dict={x: next_batch(batched_data, batch_size)[0]})
        f.write(np.array2string(w, separator=','))
        f.close()

        print('Saving...')
        saver.save(sess, './my-model.ckpt')
        print('Saving complete.')

    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=10,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
