import tensorflow as tf
import data_reader
import conv_nn_plates_light as model
import numpy as np
import random

from datetime import datetime

NUM_OF_EPOCHS = 20
BATCH_SIZE = 5
LEARNING_RATE = 1e-4

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train_classificator.txt'


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


def get_loss(y, y_):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    return loss


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, y, fc1_dropout_prob = model.get_classification_model()

    y_ = tf.placeholder(tf.float32, [None, model.OutputClasses], name='losses')

    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    print('Data batching...')

    filename_queue = load_full_image_list(TRAIN_FILE_PATH)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        print('Start time is', datetime.today())

        image_count = len(filename_queue)
        iters_count = image_count * num_of_epochs

        for epoch in range(0, num_of_epochs):
            batched_data = get_batched_data(filename_queue, batch_size)
            
            for step in range(0, image_count):
                batch = next_batch(batched_data, step % len(batched_data[0]))
                try:
                    train_step.run(feed_dict={x: batch[0], y_: batch[1], fc1_dropout_prob: 0.5})
                except ValueError:
                    print('ValueError in file:', batched_data[1][step % len(batched_data[0])])

                if step % image_count == 0:
                    print('Epoch', int(step / image_count + 1), 'of', num_of_epochs)
                    if step != 0:
                        print('Saving...')
                        saver.save(sess, './classification-model')
                        print('Saving complete.')
                else:
                    if step % 100 == 0 and step != 0:
                        print('Step', step, 'of', iters_count)
                        if step % 1000 == 0:
                            print('Saving...')
                            saver.save(sess, './classification-model')
                            print('Saving complete.')

        print('Saving...')
        saver.save(sess, './classification-model')
        print('Saving complete.')

    print('End time is', datetime.today())
    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=NUM_OF_EPOCHS,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
