from Structured.base.base_train import BaseTrain
from Structured.data_loader.random_image import generate_random_image
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import os


class PlateNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(PlateNetTrainer, self).__init__(sess, model, data, config, logger)

        self.num_iter_per_epoch = self.data.num_batches


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """

        losses = []
        # accs = []

        loop = tqdm(range(self.num_iter_per_epoch))

        for _ in loop:
            # loss, acc = self.train_step()
            loss = self.train_step()

            losses.append(loss)
            # accs.append(acc)

        loop.close()

        mean_loss = np.min(losses)
        # mean_acc = np.mean(accs)

        print("\nloss:", mean_loss)
        # print("\naccuracy:", mean_acc)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {
            'loss': mean_loss,
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        self.model.saver.save(
            self.sess, os.path.join(
                self.config.checkpoint_dir, self.config.exp_name
            )
        )

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        H, W, C = self.config.input_shape

        b_imgs, b_boxes = next(self.data.next_batch())

        b_imgs = b_imgs / 255.
        b_boxes = b_boxes / (0.5 * W, 0.5 * H, 0.5 * W, 0.5 * H) - 1.
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
            self.model.inputs_tensor: b_imgs,
            self.model.gt_boxes: b_boxes,
        }

        self.model.optimizer.run(feed_dict=feed_dict)

        loss = self.model.loss.eval(feed_dict=feed_dict)

        return loss
