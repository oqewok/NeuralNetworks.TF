import tensorflow as tf
import data_reader
import conv_nn_plates_light as model
import numpy as np
import random

from datetime import datetime

NUM_OF_EPOCHS = 30
BATCH_SIZE = 20
LEARNING_RATE = 1e-3

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train.txt'


# Загружает список изображений и соответствующих им меток
def load_full_image_list(filename):
    filename_queue = data_reader.read_labeled_image_list(filename)
    return filename_queue


# Загрузка самих изображений и меток из файлов
def load_train_data(image_files, labels_files):
    # здесь берем порциями данные
    data = data_reader.read_images_from_disk(image_files, labels_files)
    return data


# Загружает список изображений и меток и дробит данные на равные части.
def get_batched_data(filename_queue, batch_size):
    images = list(filename_queue.keys())

    random.shuffle(images)

    labels = [filename_queue[image] for image in images]

    batched_img = [images[i:i + batch_size] for i in range(0, len(filename_queue), batch_size)]
    batched_masks = [labels[i:i + batch_size] for i in range(0, len(filename_queue), batch_size)]
    return [batched_img, batched_masks]


# Загружает порцию изображений и меток из файлов
def next_batch(batched_data, batch_index):
    i, l = batched_data[0], batched_data[1]
    img_batch, lab_batch = i[batch_index], l[batch_index]

    images, masks = load_train_data(img_batch, lab_batch)
    return [images, masks]


def get_loss(prediction, y):
    loss = tf.sqrt(tf.nn.l2_loss(y - prediction))

    return loss


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, prediction = model.get_classification_model()

    y = tf.placeholder(tf.float32, [None, model.OutputNodesCount], name='losses')

    loss = get_loss(prediction, y)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    print('Data batching...')

    filename_queue = load_full_image_list(TRAIN_FILE_PATH)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        print('Start time is', datetime.today())

        epoch_loss = 0.0

        for epoch in range(0, num_of_epochs):
            print('Epoch', int(epoch + 1), 'of', num_of_epochs, 'loss:', epoch_loss)
            if epoch != 0:
                print('Saving...')
                saver.save(sess, './model')
                print('Saving complete.')
            if epoch == 10:
                learn_rate = 0.0001
            if epoch == 20:
                learn_rate = 0.00001
            epoch_loss = 0.0
            batched_data = get_batched_data(filename_queue, batch_size)

            for step in range(0, len(batched_data[0])):
                batch = next_batch(batched_data, step % len(batched_data[0]))
                try:
                    _, c = sess.run([optimizer, loss], feed_dict={x: batch[0], y: batch[1]})
                    if c > epoch_loss:
                        epoch_loss = c
                except ValueError:
                    print('ValueError in file:', batched_data[1][step % len(batched_data[0])])

                if step % batch_size == 0 and step != 0:
                    print('Step', step, 'of', len(batched_data[0]))

        print('Saving...')
        saver.save(sess, './model')
        print('Saving complete.')

    print('End time is', datetime.today())
    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=NUM_OF_EPOCHS,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
