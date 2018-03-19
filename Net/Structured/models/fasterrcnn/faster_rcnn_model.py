from Structured.base.base_model import BaseModel
from Structured.nets.vgg16 import build_basic_vgg16
from Structured.models.fasterrcnn.rpn import RPN

import tensorflow as tf


class FasterRCNNModel(BaseModel):
    def __init__(self, config):
        super(FasterRCNNModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you predict the tensorflow graph of any model you want and also define the loss.

        # Value of training mode
        self.is_training = self.config.is_training

        # Inputs or X. Tensor for the batch of images.
        self.inputs_tensor      = None

        # Tensor for training mode description. If true => training mode, else => evaluation mode.
        self.is_training_tensor = None

        # convolution features after basic CNN feature extraction
        self.conv_feats_tensor  = None

        # convolution features shape (ndarray).
        self.conv_feats_shape   = None

        # Build basic CNN for feature extraction. Default is VGG16.
        ''' For example:
        elif self.config.basic_cnn == "resnet50":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_resnet50(
            self.config)'''
        if self.config.basic_cnn == "vgg16":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)
        else:
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)

        # Build RPN
        #TODO: Task 8: Передать gt_boxes в конструктор
        self.rpn = RPN(
            self.config, self.conv_feats_tensor, self.conv_feats_shape, gt_boxes=None, is_training=self.is_training_tensor
        )

        self.rpn_cls_target, self.rpn_bbox_target, self.rpn_max_overlap = self.rpn.predict()
        pass

    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass


from Structured.utils.config import process_config
config = process_config("E:/Study/Mallenom/NeuralNetworks.TF/Net/Structured/configs/fastercnn.json")
model = FasterRCNNModel(config)
pass
