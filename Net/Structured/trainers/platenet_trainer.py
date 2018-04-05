import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pylab import rcParams
from tqdm import tqdm

from Structured.base.base_train import BaseTrain
from Structured.utils.img_preproc import *


class PlateNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(PlateNetTrainer, self).__init__(sess, model, data, config, logger)

        self.num_iter_per_epoch = self.data.num_batches
        H, W, C = self.config.input_shape

        self.X_test, self.Y_test = self.data.next_batch(self.data.num_test, "TEST")
        #self.X_test = subtract_channels(self.X_test)
        self.X_test = self.X_test / 255.
        #self.Y_test = self.Y_test / (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H, 1) - 1.
        #self.Y_test[:, 4] = self.Y_test[:, 4] + 1.

        self.best_loss = 1000000000
        self.best_cls_loss = 1000000000
        pass

    # def LoadData(self, paths):
    #     xs = []
    #     ys = []
    #     for ex_paths in paths:
    #         img_path = ex_paths[0]
    #         ann_path = ex_paths[1]
    #         xs.append(self.LoadImage(img_path))
    #         ys.append(self.LoadAnnotation(ann_path))
    #
    #     return np.array(xs), np.array(ys)
    #
    #
    # def LoadImage(self, fname):
    #     img = io.imread(fname)
    #     img = resize_img(img, self.config.input_shape, as_int=True) / 255.
    #     return img

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """

        losses = []
        accs = []
        
        loop = tqdm(range(self.num_iter_per_epoch))

        for _ in loop:
            cls_loss, acc = self.train_step()

            losses.append(cls_loss)
            accs.append(acc)
            
            if cls_loss < self.best_loss:
               # self.best_loss = bbox_loss
                self.best_loss = cls_loss
                self.model.saver.save(
                    self.sess, os.path.join(
                        self.config.checkpoint_dir, self.config.exp_name
                    )
                )

        loop.close()

        mean_acc = np.mean(accs)

        print("\naccuracy:", mean_acc)
        print("\ncls loss:", self.best_loss)
       # print("bbox loss:", self.best_loss)

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        H, W, C = self.config.input_shape

        b_x, b_y = self.data.next_batch(self.config.batch_size, "TRAIN")

        #b_x = subtract_channels(b_x)
        b_x = b_x / 255.
        #b_y = b_y / (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H, 1) - 1.
        #b_y[:, 4] = b_y[:, 4] + 1.
        # # generate random image for negative samples
        # neg_imgs = []
        # for i in range(len(img)):
        #     neg_img = generate_random_image(self.config.input_shape)
        #     neg_imgs.append(neg_img)
        #
        # neg_imgs = np.array(neg_imgs)
        # imgs = np.concatenate((img, neg_imgs), axis=0)
        #
        # ones = np.ones(shape=[len(img)], dtype=np.int32)
        # zeros = np.zeros(shape=[len(img)], dtype=np.int32)
        # labels = np.concatenate((ones, zeros))
        #
        # indices = np.random.permutation(len(ones) + len(zeros))
        #
        # b_img       = imgs[indices]
        # b_boxes    = labels[indices]

        # graph = tf.get_default_graph()
        # inputs = graph.get_tensor_by_name('truediv:0')

        feed_dict = {
            self.model.inputs_tensor: b_x,
            self.model.gt_boxes: b_y,
            self.model.is_train: True
        }

        self.sess.run(self.model.optimizer, feed_dict=feed_dict)

        cls_loss = self.model.loss.eval(feed_dict={
            self.model.inputs_tensor: self.X_test,
            self.model.gt_boxes: self.Y_test,
            self.model.is_train: False,
        })
        '''bbox_loss = self.model.bbox_loss.eval(feed_dict={
            self.model.inputs_tensor: self.X_test,
            self.model.gt_boxes: self.Y_test,
            self.model.is_train: False,
        })'''
        acc = self.model.accuracy.eval(feed_dict={
            self.model.inputs_tensor: self.X_test,
            self.model.gt_boxes: self.Y_test,
            self.model.is_train: False,
        })
        return cls_loss, acc


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
