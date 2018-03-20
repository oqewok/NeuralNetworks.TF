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
        losses = []
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for i in loop:
            loss = self.train_step()
            losses.append(loss)

        mean_loss = np.mean(losses)
        print("Epoch loss = ", mean_loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {}
        summaries_dict['loss'] = mean_loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        # TODO: fix: ValueError: 'latest_filename' collides with 'save_path': 'checkpoint' and '../experiments\faster-rcnn\checkpoint'
        #self.model.saver.save(self.sess, self.config.checkpoint_dir)


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

        _, loss = self.sess.run(
            [self.model.optimizer, self.model.loss], feed_dict=feed_dict
        )

        return loss
