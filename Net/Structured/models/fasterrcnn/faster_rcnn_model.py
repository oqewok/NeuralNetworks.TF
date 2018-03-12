from Structured.base.base_model import BaseModel
from Structured.nets.vgg16 import build_basic_vgg16
import tensorflow as tf


class FasterRCNNModel(BaseModel):
    def __init__(self, config):
        super(FasterRCNNModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.

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
        elif self.config.basic_cnn == "ResNet50":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_resnet50(
            self.config)'''
        if self.config.basic_cnn == "VGG16":
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)
        else:
            self.inputs_tensor, self.is_training_tensor, self.conv_feats_tensor, self.conv_feats_shape = build_basic_vgg16(
                self.config)

        pass

    def init_saver(self):
        #here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass

'''
from Structured.utils.config import process_config
config = process_config("E:/Study/Mallenom/NeuralNetworks.TF/Net/Structured/configs/fastercnn.json")
model = FasterRCNNModel(config)
pass
'''