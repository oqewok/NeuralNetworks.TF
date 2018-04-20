from Structured.base.base_train import BaseTrain
from Structured.utils.bbox_overlap import bbox_overlap
from tqdm import tqdm

import numpy as np
import os


class FasterRCNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(FasterRCNNTrainer, self).__init__(sess, model, data, config, logger)

        self.num_iter_per_epoch = self.data.num_train
        self.best_loss = 100000000000
        self.best_rpn_loss = 100000000000
        self.best_cls_loss = 100000000000


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        rpn_cls_losses = []
        rpn_reg_losses = []

        rcnn_cls_losses = []
        rcnn_reg_losses = []

        regularization_losses = []

        total_losses = []
        loop = tqdm(range(self.num_iter_per_epoch))

        rpn_cls_loss = 0
        rpn_reg_loss = 0
        rcnn_cls_loss = 0
        rcnn_reg_loss = 0
        total_loss = 0
        regularization_loss = 0

        for i in loop:
            if self.config.with_rcnn:
                rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, regularization_loss = self.train_step()
                total_loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss + regularization_loss
            else:
                rpn_cls_loss, rpn_reg_loss, regularization_loss = self.train_step()
                total_loss = rpn_cls_loss + rpn_reg_loss + regularization_loss

            rpn_cls_losses.append(rpn_cls_loss)
            rpn_reg_losses.append(rpn_reg_loss)
            regularization_losses.append(regularization_loss)

            if self.config.with_rcnn:
                rcnn_cls_losses.append(rcnn_cls_loss)
                rcnn_reg_losses.append(rcnn_reg_loss)

            total_losses.append(total_loss)

        loop.close()

        mean_rpn_cls_loss = np.mean(rpn_cls_losses)
        mean_rpn_reg_loss = np.mean(rpn_reg_losses)

        mean_regularization_loss = np.mean(regularization_losses)

        mean_rcnn_cls_loss = np.mean(rcnn_cls_losses)
        mean_rcnn_reg_loss = np.mean(rcnn_reg_losses)

        mean_total_loss = np.mean(total_losses)

        print("\nrpn cls loss:", mean_rpn_cls_loss)
        print("\nrpn reg loss:", mean_rpn_reg_loss)

        if self.config.with_rcnn:
            print("\nrcnn cls loss:", mean_rcnn_cls_loss)
            print("\nrcnn reg loss:", mean_rcnn_reg_loss)

        print("\ntotal loss:", mean_total_loss)
        print("\n regularization loss:", mean_regularization_loss)
        print("\n total - regularization loss:", mean_total_loss - mean_regularization_loss)

        if self.config.with_rcnn:
            if mean_rpn_cls_loss + mean_rpn_reg_loss + mean_rcnn_cls_loss + mean_rcnn_reg_loss < self.best_loss:
                self.best_loss = mean_rpn_cls_loss + mean_rpn_reg_loss + mean_rcnn_cls_loss + mean_rcnn_reg_loss

                self.model.saver.save(
                    self.sess, os.path.join(
                        self.config.checkpoint_dir, self.config.exp_name
                    )
                )
        else:
            if mean_rpn_cls_loss + mean_rpn_reg_loss  < self.best_loss:
                self.best_loss = mean_rpn_cls_loss + mean_rpn_reg_loss

                self.model.saver.save(
                    self.sess, os.path.join(
                        self.config.checkpoint_dir, self.config.exp_name
                    )
                )

            pb_file = os.path.join(
                self.config.checkpoint_dir,
                "fasterrcnn.pb")

            with open(pb_file, 'wb') as f:
                f.write(self.sess.graph_def.SerializeToString())


        print(print("\nbest loss:", self.best_loss))
        # summaries_dict = {
        #     'rpn_cls_loss': mean_rpn_cls_loss,
        #     'rpn_reg_losses': mean_rpn_reg_loss,
        #     'total_loss': mean_total_loss
        # }
        #
        # if self.config.with_rcnn:
        #     summaries_dict.update({
        #         'rcnn_cls_losses': mean_rcnn_cls_loss,
        #         'rcnn_reg_losses': mean_rcnn_reg_loss,
        #     })
        #
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        # self.model.saver.save(
        #     self.sess, os.path.join(
        #         self.config.checkpoint_dir, self.config.exp_name
        #     )
        # )

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        img, boxes = self.data.next_img()

        #img = preprocess(img)

        feed_dict = {
            self.model.inputs_tensor: [img],
            self.model.gt_boxes: boxes,
            self.model.is_training_tensor: self.model.is_training,
        }

        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "1080Ti:8908")
        if self.config.with_rcnn:
            _, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, regularization_loss = self.sess.run(
                [
                    self.model.optimizer,
                    self.model.rpn_cls_loss,
                    self.model.rpn_reg_loss,
                    self.model.rcnn_cls_loss,
                    self.model.rcnn_reg_loss,
                    self.model.regularization_loss,
                ],
                feed_dict=feed_dict
            )

            return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, regularization_loss
        else:
            _, rpn_cls_loss, rpn_reg_loss, regularization_loss = self.sess.run(
                [
                    self.model.optimizer,
                    self.model.rpn_cls_loss,
                    self.model.rpn_reg_loss,
                    self.model.regularization_loss,
                ],
                feed_dict=feed_dict
            )

            return rpn_cls_loss, rpn_reg_loss, regularization_loss


    def valid(self):
        X_val, Y_val = self.data.X_val, self.data.Y_val

        y = self.sess.graph_def.get_tensor_by_name('BoundingBoxTransform/clip_bboxes_1/concat:0')
        probs = self.sess.graph_def.get_tensor_by_name('nms/gather_nms_proposals_scores:0')
        losses = []
        ious = []

        for _ in tqdm(range(len(X_val))):
            img, boxes = self.data.next_img()

            feed_dict = {
                self.model.inputs_tensor: [img],
                self.model.gt_boxes: boxes,
                self.model.is_training_tensor: self.model.is_training,
            }

            boxes, p, loss, regularization_loss = self.sess.run(
                [
                    y, probs,
                    self.model.loss,
                    self.model.regularization_loss,
                ],
                feed_dict=feed_dict
            )

            loss = loss - regularization_loss
            losses.append(loss)

            b = boxes[p >= 0.5]
            iou = bbox_overlap(Y_val, b)
            ious.append(np.mean(iou))

        print("test loss:", np.mean(losses))
        print("mAP", np.mean(ious))
