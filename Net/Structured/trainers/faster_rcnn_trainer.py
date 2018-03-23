from Structured.base.base_train import BaseTrain
from tqdm import tqdm

from tensorflow.python import debug as tf_debug

import numpy as np
import os


class FasterRCNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(FasterRCNNTrainer, self).__init__(sess, model, data, config, logger)

        self.num_iter_per_epoch = self.data.num_train


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        losses = []
        loop = tqdm(range(self.num_iter_per_epoch))

        for i in loop:
            loss = self.train_step()
            losses.append(loss)

        loop.close()

        mean_loss = np.mean(losses)
        print("Epoch loss = ", mean_loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {}
        summaries_dict['loss'] = mean_loss
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
        img, boxes = self.data.next_img() # ?????
        feed_dict = {
            self.model.inputs_tensor: [img],
            self.model.gt_boxes: boxes,
            self.model.is_training_tensor: self.model.is_training,
        }

        #self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "1080Ti:8908")

        _, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss = self.sess.run(
            [self.model.optimizer, self.model.rpn_cls_loss, self.model.rpn_reg_loss, self.model.rcnn_cls_loss, self.model.rcnn_reg_loss],
            feed_dict=feed_dict
        )

        loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss

        return loss
