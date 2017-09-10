import tensorflow as tf
import data_reader
import conv_nn_plates_light as model
import numpy as np
import random
import image_proc

from datetime import datetime
from skimage import novice

NUM_OF_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

TRAIN_FILE_PATH = 'E:/Study/Mallenom/train.txt'
TEST_FILE_PATH = 'E:/Study/Mallenom/test.txt'
MODEL_FOLDER = './models/test_model2/'


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
    loss = tf.nn.l2_loss(y - prediction)
    return tf.sqrt(loss)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle

    # if not intersects
    if not (boxA[0] < boxB[2] and boxA[2] > boxB[0] and boxA[3] > boxB[1] and boxA[1] < boxB[3]):
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    if interArea < 0:
        interArea = 0

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


def test(test_filenames_queue, x, prediction, dropout, epoch_num):
    test_batch = get_batched_data(test_filenames_queue, 1)
    correct_predictions = 0.0

    with open('%sepoch_%d_test_log.log' % (MODEL_FOLDER, epoch_num), 'w') as logfile:
        for step in range(0, len(test_batch[0])):
            batch = next_batch(test_batch, step)
            predicted = prediction.eval(feed_dict={x: batch[0], dropout: 1.0})[0]
            original = batch[1][0]

            picture = novice.open(test_batch[0][step][0])
            width = picture.width
            height = picture.height

            predicted_decoded = image_proc.decode_rect(predicted, width, height)
            original_decoded = image_proc.decode_rect(original, width, height)

            iou = bb_intersection_over_union(predicted_decoded, original_decoded)

            log = "filename=%s; predicted=%s; predicted_decoded=%s; original=%s; original_decoded=%s IoU=%f;\n" % (
                test_batch[0][step],
                np.array2string(predicted, separator=','),
                np.array2string(predicted_decoded, separator=','),
                np.array2string(original, separator=','),
                np.array2string(original_decoded, separator=','),
                iou)
            logfile.write(log)

            if iou >= 0.5:
                correct_predictions += 1.0

    accuracy = correct_predictions / len(test_batch[0])
    return accuracy


def train(num_of_epochs, learn_rate, batch_size):
    np.set_printoptions(threshold=np.nan, suppress=True)

    print('Loading model...')
    x, prediction, dropout = model.get_classification_model()

    y = tf.placeholder(tf.float32, [None, model.OutputNodesCount], name='losses')

    loss = get_loss(prediction, y)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    print('Data batching...')

    train_filename_queue = load_full_image_list(TRAIN_FILE_PATH)
    test_filenames_queue = load_full_image_list(TEST_FILE_PATH)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        print('Training...')
        sess.run(init)

        print('Start time is', datetime.today())
        accuracy = test(test_filenames_queue, x, prediction, dropout, 0)
        epoch_loss = 0.0

        for epoch in range(0, num_of_epochs):
            print('Epoch', int(epoch + 1), 'of', num_of_epochs, 'loss:', epoch_loss)

            if epoch != 0:
                print('Saving...')
                saver.save(sess, '%s/model' % MODEL_FOLDER)
                print('Saving complete.')

                print('Testing...')
                accuracy = test(test_filenames_queue, x, prediction, dropout, epoch)
                print('Accuracy:', accuracy)

            epoch_loss = 0.0
            batched_data = get_batched_data(train_filename_queue, batch_size)

            for step in range(0, len(batched_data[0])):
                batch = next_batch(batched_data, step % len(batched_data[0]))
                try:
                    _, c = sess.run([optimizer, loss], feed_dict={x: batch[0], y: batch[1], dropout: 0.5})
                    epoch_loss += c
                except ValueError:
                    print('ValueError in file:', batched_data[1][step % len(batched_data[0])])

                if step % batch_size == 0 and step % 100 == 0:
                    print('Step', step, 'of', len(batched_data[0]))

            epoch_loss /= len(batched_data[0])

        print('Result loss:', epoch_loss)
        print('Saving...')
        saver.save(sess, '%s/model' % MODEL_FOLDER)
        print('Saving complete.')

        print('Testing...')
        accuracy = test(test_filenames_queue, x, prediction, dropout, num_of_epochs)
        print('Accuracy:', accuracy)

    print('End time is', datetime.today())
    print('Success!')

if __name__ == "__main__":
    train(num_of_epochs=NUM_OF_EPOCHS,
          learn_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE)
