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
        accs = []

        loop = tqdm(range(self.num_iter_per_epoch))

        for _ in loop:
            loss, acc = self.train_step()

            losses.append(loss)
            accs.append(acc)

        loop.close()

        mean_loss = np.mean(losses)
        mean_acc = np.mean(accs)

        print("\nloss:", mean_loss)
        print("\naccuracy:", mean_acc)

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
        img = next(self.data.next_batch())

        # generate random image for negative samples
        neg_imgs = []
        for i in range(len(img)):
            neg_img = generate_random_image(self.config.input_shape)
            neg_imgs.append(neg_img)

        neg_imgs = np.array(neg_imgs)
        imgs = np.concatenate((img, neg_imgs), axis=0)

        ones = np.ones(shape=[len(img)], dtype=np.int32)
        zeros = np.zeros(shape=[len(img)], dtype=np.int32)
        labels = np.concatenate((ones, zeros))

        indices = np.random.permutation(len(ones) + len(zeros))

        b_img       = imgs[indices]
        b_labels    = labels[indices]

        # graph = tf.get_default_graph()
        # inputs = graph.get_tensor_by_name('truediv:0')

        feed_dict = {
            self.model.inputs_tensor: b_img,
            self.model.labels: b_labels,
            self.model.is_training_tensor: self.model.is_training,
        }

        _, loss, acc, pred, out = self.sess.run(
                [
                    self.model.optimizer,
                    self.model.loss,
                    self.model.accuracy,
                    self.model.correct_prediction,
                    self.model.outputs,
                ],
                feed_dict=feed_dict
            )

        return loss, acc
