from Structured.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class FasterRCNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(FasterRCNNTrainer, self).__init__(sess, model, data, config, logger)


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for i in loop:
            loss = self.train_step()

            if i % 100:
                print("loss = " + loss)


        self.model.saver.save(self.sess, self.config.checkpoint_dir)


    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        img, gt_boxes = next(self.data.next_batch())  # ?????
        feed_dict = {
            self.model.inputs_tensor: img,
            self.model.gt_boxes: gt_boxes,
        }

        _, loss = self.sess.run(
            [self.model.optimizer, self.model.loss], feed_dict=feed_dict
        )

        return loss
