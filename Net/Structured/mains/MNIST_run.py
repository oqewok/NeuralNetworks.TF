import tensorflow as tf

from Structured.data_loader.mnist_data_provider import MnistDataProvider
from Structured.models.mnist_model import MnistModel
from Structured.trainers.mnist_trainer import MnistTrainer
from Structured.utils.config import process_config
from Structured.utils.dirs import create_dirs
from Structured.utils.logger import Logger
from Structured.utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    global config
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()

    #sess.run(tf.global_variables_initializer())

    # create instance of the model you want
    model = MnistModel(config)
    # create your data generator
    data = MnistDataProvider(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = MnistTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
