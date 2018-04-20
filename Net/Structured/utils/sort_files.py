from Structured.data_loader.art_car_plates_data_provider import ArtificalCarPlatesDataProvider
from Structured.data_loader.reader import Reader
from Structured.utils.config import process_config
from skimage import io
import matplotlib.pyplot as plt
from Structured.data_loader.parser import MarkupParser


import collections
import numpy as np
import os

config = process_config(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\fastercnn.json")

data = ArtificalCarPlatesDataProvider(config)

X, Y = data.samples["TRAIN"][0], data.samples["TRAIN"][1]
parser = MarkupParser()

dir = config.train_root_directory

for i in range(len(X)):
    country = parser.getCountry(Y[i])

    new_path = os.path.join(dir, country)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    os.rename(X[i], os.path.join(new_path, os.path.basename(X[i])))
    os.rename(Y[i], os.path.join(new_path, os.path.basename(Y[i])))

