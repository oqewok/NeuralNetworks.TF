from Structured.utils.config import process_config
from Structured.data_loader.reader import Reader
import os

# Формирует соответствующие txt-файлы с выборкой из указанных директорий в config-файле.
def make_samples_file(config_file, type):
    config = process_config(config_file)

    # TODO: Task5: Придумать куда сохранять txt-файлы
    if not os.path.exists(config.train_files_directory):
        os.mkdir(config.train_files_directory)

    if type == "xml":
        Reader.get_samples_file(config.train_root_directory, os.path.abspath(
            os.path.join(config.train_files_directory, "train.txt")))
        Reader.get_samples_file(config.valid_root_directory, os.path.abspath(
            os.path.join(config.train_files_directory, "valid.txt")))
        Reader.get_samples_file(config.test_root_directory,  os.path.abspath(
            os.path.join(config.train_files_directory, "test.txt")))
    elif type == "json":
        Reader.get_samples_file_json(config.train_root_directory, os.path.abspath(
            os.path.join(config.train_files_directory, "train.txt")))
        # Reader.get_samples_file_json(config.valid_root_directory, os.path.abspath(
        #     os.path.join(config.train_files_directory, "valid.txt")))
        Reader.get_samples_file_json(config.test_root_directory, os.path.abspath(
            os.path.join(config.train_files_directory, "test.txt")))
    else:
        raise ValueError("Wrong parser type")


if __name__ == '__main__':
    make_samples_file(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\configs\\fastercnn.json",
        "xml"
    )
