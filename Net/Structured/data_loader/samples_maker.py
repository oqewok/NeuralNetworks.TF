from Structured.utils.config import process_config
from Structured.data_loader.reader import Reader


# Формирует соответствующие txt-файлы с выборкой из указанных директорий в config-файле.
def make_samples_file(config_file):
    config = process_config(config_file)

    # TODO: Task5: Придумать куда сохранять txt-файлы
    Reader.get_samples_file(config.train_root_directory, "train.txt")
    Reader.get_samples_file(config.valid_root_directory, "valid.txt")
    Reader.get_samples_file(config.test_root_directory,  "test.txt")
