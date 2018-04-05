import random
import time
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
from pylab import rcParams
import os
import json

from Structured.utils.img_preproc import *

MODEL_PATH = 'C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\experiments\\platenet_test_3'
SAMPLES_PATHS = ['C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\Licence_plates_artificial\\']

EPOCH = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 20
INPUT_SHAPE = [64, 128, 3]
H, W, C = INPUT_SHAPE

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def LoadImage(fname):
    img = io.imread(fname)
    img = resize_img(img, INPUT_SHAPE, as_int=True) / 255.
    return img


def LoadAnnotation(fname):
    with open(fname) as data_file:
        data = json.load(data_file)

    points = np.array(data["objects"][0]["points"]["exterior"])

    x, y = np.split(points, 2, axis=1)

    left = np.min(x)
    top = np.min(y)
    right = np.max(x)
    bottom = np.max(y)

    return [left, top, right, bottom]


def ReadDirFiles(dname):
    paths = []
    for file in os.listdir(os.path.join(dname, "img")):
        bname = os.path.basename(file).split(".")[0]

        img_name = os.path.join(dname, "img", file)
        ann_name = os.path.join(dname, "ann", bname + ".json")
        paths.append((img_name, ann_name))
    return paths


def ReadPaths(paths):
    all_paths = []
    for path in paths:
        temp_paths = ReadDirFiles(path)
        all_paths.extend(temp_paths)
    return all_paths


def get_tags(fname):
    with open(fname) as data_file:
        data = json.load(data_file)
    tags = data["tags"]
    return tags


def train_test_split(paths, train_tag="train", test_tag="test"):
    train_paths = paths[0:9819]
    test_paths = paths[9819:]

    return train_paths, test_paths


def LoadData(paths):
    xs = []
    ys = []
    for ex_paths in paths:
        img_path = ex_paths[0]
        ann_path = ex_paths[1]
        xs.append(LoadImage(img_path))
        ys.append(LoadAnnotation(ann_path))

    return np.array(xs), np.array(ys)


all_paths = ReadPaths(SAMPLES_PATHS)
tr_paths, te_paths = train_test_split(all_paths)

print(len(tr_paths))
print(len(te_paths))

X_train, Y_train = LoadData(tr_paths)
X_test, Y_test = LoadData(te_paths)

print("check shapes:")
print("X_train - ", X_train.shape)
print("Y_train - ", Y_train.shape)
print("X_test - ", X_test.shape)
print("Y_test - ", Y_test.shape)


def show_image(image, labels, gt=None):
    plt.imshow(image)
    gca = plt.gca()

    if gt is not None:
        rect = Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], edgecolor='b',
                         fill=False)
        gca.add_patch(rect)

    rect = Rectangle((labels[0], labels[1]), labels[2] - labels[0], labels[3] - labels[1], edgecolor='r', fill=False)
    gca.add_patch(rect)


def plot_images(images, labels, gt=None):
    rcParams['figure.figsize'] = 14, 8
    plt.gray()
    fig = plt.figure()
    for i in range(min(9, images.shape[0])):
        fig.add_subplot(3, 3, i + 1)
        show_image(images[i], labels[i], gt[i])
    plt.show()


xs = [random.randint(0, X_train.shape[0] - 1) for _ in range(9)]
plot_images(X_train[xs], Y_train[xs], gt=Y_train[xs])


class Dataset:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = X.shape[0]

    def next_batch(self, batch_size=20):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.X[start:end], self.Y[start:end]


    def epoch_completed(self):
        return self._epochs_completed


def mse(expected, predicted):
    se = tf.square(expected - predicted)
    return tf.reduce_mean(se)

def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


PIXEL_COUNT = X_train.shape[1] * X_train.shape[2]
LABEL_COUNT = Y_train.shape[1]


# Create placeholders for image data and expected point positions

class Model(object):

    def __init__(self, x_placeholder, y_placeholder, dropout_prob, output):
        xxx = 0
        self.x_placeholder = x_placeholder
        self.y_placeholder = y_placeholder
        self.dropout_prob = dropout_prob
        self.output = output

# Build neural network
def build_model():
    x_placeholder = tf.placeholder(tf.float32, shape=[None, H, W, C])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
    dropout_prob  = tf.placeholder(tf.float32)

    # Convolution Layer 1
    W_conv1 = weight_variable("w1", [3, 3, C, 32])
    b_conv1 = bias_variable("b1", [32])
    h_conv1 = tf.nn.relu(conv2d(x_placeholder, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Convolution Layer 2
    W_conv2 = weight_variable("w2", [2, 2, 32, 64])
    b_conv2 = bias_variable("b2", [64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # Convolution Layer 3
    W_conv3 = weight_variable("w3", [2, 2, 64, 128])
    b_conv3 = bias_variable("b3", [128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # Dense layer 1
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 16 * 128])
    W_fc1 = weight_variable("w4", [8 * 16 * 128, 500])
    b_fc1 = bias_variable("b4", [500])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, dropout_prob)
    # Dense layer 2
    W_fc2 = weight_variable("w5", [500, 500])
    b_fc2 = bias_variable("b5", [500])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # Output layer
    W_out = weight_variable("w6", [500, LABEL_COUNT])
    b_out = bias_variable("b6", [LABEL_COUNT])

    output = tf.matmul(h_fc2, W_out) + b_out

    model = Model(x_placeholder, y_placeholder, dropout_prob, output)

    return model


#X2_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X2_train = X_train
Y2_train = Y_train / (0.5*W, 0.5*H, 0.5*W, 0.5*H) - 1.0

#X2_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X2_test = X_test
Y2_test = Y_test / (0.5*W, 0.5*H, 0.5*W, 0.5*H) - 1.0

dataset = Dataset(X2_train, Y2_train)

g = tf.Graph()
with g.as_default():
    model = build_model()
    loss = mse(model.y_placeholder, model.output)

    saver = tf.train.Saver()
    start_time = time.time()
    best_score = 1

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    #saver.restore(session, os.path.join(MODEL_PATH, "model"))

    last_epoch = -1
    while dataset.epoch_completed() < EPOCH:
        (batch_x, batch_y) = dataset.next_batch(BATCH_SIZE)
        _ = session.run(train_step, feed_dict={
            model.x_placeholder: batch_x,
            model.y_placeholder: batch_y,
            model.dropout_prob: 0.5
        })
        if dataset.epoch_completed() > last_epoch:
            last_epoch = dataset.epoch_completed()
            score_test = loss.eval(feed_dict={
                model.x_placeholder: X2_test,
                model.y_placeholder: Y2_test,
                model.dropout_prob: 1.0
            })
            if score_test < best_score:
                best_score = score_test
                saver.save(session, os.path.join(MODEL_PATH, "model"))
            if dataset.epoch_completed() % 1 == 0:
                epm = 60 * dataset.epoch_completed() / (time.time() - start_time)
                print('Epoch: %d, Score: %f, Epoch per minute: %f' % (dataset.epoch_completed(), score_test, epm))
    print('Finished in %f seconds.' % (time.time() - start_time))

    session.close()

g = tf.Graph()
with g.as_default():
    session = tf.InteractiveSession()
    model = build_model()
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(MODEL_PATH, "model"))
    ids = [random.randint(0, X2_test.shape[0] - 1) for _ in range(9)]
    predictions = model.output.eval(session=session, feed_dict={
        model.x_placeholder: X2_test[ids],
        model.dropout_prob: 1.0
    })
    plot_images(X_test[ids], (predictions + 1) * (0.5*W, 0.5*H, 0.5*W, 0.5*H), gt=Y_test[ids])
    session.close()

g = tf.Graph()
with g.as_default():
    session = tf.InteractiveSession()
    model = build_model()
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(MODEL_PATH, "model"))

    ids = [random.randint(0, X_train.shape[0] - 1) for _ in range(9)]
    predictions = model.output.eval(session=session, feed_dict={
        model.x_placeholder: X2_train[ids],
        model.dropout_prob: 1.0
    })
    plot_images(X_train[ids], (predictions + 1) * (0.5*W, 0.5*H, 0.5*W, 0.5*H), gt=Y_train[ids])

    session.close()
