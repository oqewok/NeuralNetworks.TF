import tensorflow as tf
import numpy as np
import data_reader
import conv_nn_faces_model as model
import sys

NUM_OF_EPOCHS = 10000
BATCH_SIZE = 10
LEARNING_RATE = 1e-4


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def load_train_data():
    # здесь берем порциями данные
    names, labels = data_reader.read_labeled_image_list()
    data = data_reader.read_images_from_disk(names, labels)
    return data


def get_batched_data(data, BATCH_COUNT):
    img, masks = data[0], data[1]
    batched_img = [img[i:i + BATCH_COUNT] for i in range(0, len(img), BATCH_COUNT)]
    batched_masks = [masks[i:i + BATCH_COUNT] for i in range(0, len(masks), BATCH_COUNT)]
    return [batched_img, batched_masks]


def next_batch(batched_data, batch_index):
    i, l = batched_data[0], batched_data[1]
    images = i[batch_index, :]
    masks = l[batch_index, :]
    return [images, masks]


def get_loss(y, y_):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    return loss


def train(num_of_epochs, learn_rate, batch_size):
    x, y, params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, model.OutputNodesCount])

    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    data = load_train_data()
    batched_data = get_batched_data(data, batch_size)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_of_epochs):
            batch = next_batch(batched_data, i % batch_size)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('Success!')


if __name__ == "__main__":
    train(num_of_epochs=75,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
