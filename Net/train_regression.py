import tensorflow as tf
import data_reader
import conv_nn_plates as model
import numpy as np

from datetime import datetime

NUM_OF_EPOCHS = 100
BATCH_SIZE = 200
LEARNING_RATE = 1e-2

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train.txt'


# Загружает список изображений и соответствующих им меток
def load_full_image_list(filename):
    names, labels = data_reader.read_labeled_image_list(filename)
    return names, labels


# Загрузка самих изображений и меток из файлов
def load_train_data(names, labels):
    # здесь берем порциями данные
    data = data_reader.read_images_from_disk(names, labels)
    return data


# Загружает список изображений и меток и дробит данные на равные части.
def get_batched_data(batch_size):
    names, labels = load_full_image_list(TRAIN_FILE_PATH)
    batched_img = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    batched_masks = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    return [batched_img, batched_masks]


# Загружает порцию изображений и меток из файлов
def next_batch(batched_data, batch_index):
    i, l = batched_data[0], batched_data[1]
    img_batch, lab_batch = i[batch_index], l[batch_index]

    images, masks = load_train_data(img_batch, lab_batch)
    return [images, masks]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_loss(y, y_):
    loss = tf.nn.l2_loss(y - y_, name='l2_loss')

    return loss


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, y, fc1_dropout_prob, conv_dropout_prob, _, l2_loss = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, model.OutputNodesCount], name='losses')

    loss = get_loss(y, y_)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    print('Data batching...')
    batched_data = get_batched_data(batch_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        print('Start time is', datetime.today())

        image_count = batched_data[0] * batch_size
        iters_count = image_count * num_of_epochs + 1

        for step in range(iters_count):
            batch = next_batch(batched_data, step % len(batched_data[0]))
            try:
                train_step.run(feed_dict={x: batch[0], y_: batch[1], fc1_dropout_prob: 0.5})
            except ValueError:
                print('ValueError in file:', batched_data[1][step % len(batched_data[0])])

            if step % image_count == 0:
                print('Epoch', step / image_count + 1, 'of', num_of_epochs)
                if step != 0:
                    print('Saving...')
                    saver.save(sess, './model')
                    print('Saving complete.')

        print('Saving...')
        saver.save(sess, './model')
        print('Saving complete.')

    print('End time is', datetime.today())
    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=NUM_OF_EPOCHS,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
