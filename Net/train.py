import tensorflow as tf
import data_reader
import conv_nn_faces_model as model
import numpy as np

from datetime import datetime

NUM_OF_EPOCHS = 66790000
BATCH_SIZE = 1
LEARNING_RATE = 1e-4


# Загружает список изображений и соответствующих им меток
def load_full_image_list():
    names, labels = data_reader.read_labeled_image_list()
    return names, labels


# Загрузка самих изображений и меток из файлов
def load_train_data(names, labels):
    # здесь берем порциями данные
    data = data_reader.read_images_from_disk(names, labels)
    return data


# Загружает список изображений и меток и дробит данные на равные части.
def get_batched_data(batch_size):
    names, labels = load_full_image_list()
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
    # loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=y[:, :1], labels=y_[:, :1]))
    return loss


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, y, keep_prob = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, model.OutputNodesCount], name='losses')

    loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    print('Data batching...')
    batched_data = get_batched_data(batch_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        print('Start time is', datetime.today())

        for step in range(num_of_epochs + 1):
            batch = next_batch(batched_data, step % len(batched_data[0]))
            try:
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            except ValueError:
                print('ValueError in file:', batched_data[1][step % len(batched_data[0])])

            if step % 100 == 0:
                print('Step', step, 'of', num_of_epochs)

                if step % 1000 == 0 and step != 0:
                    print('Saving...')
                    saver.save(sess, './model')
                    print('Saving complete.')

            # f = open('E:/Study/Mallenom/1.txt', 'w')
            # w = y.eval(feed_dict={x: next_batch(batched_data, 0)[0], keep_prob: 1.0})
            # f.write(np.array2string(w, separator=','))
            # f.close()

        print('Saving...')
        saver.save(sess, './model')
        print('Saving complete.')

    print('End time is', datetime.today())
    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=NUM_OF_EPOCHS,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
