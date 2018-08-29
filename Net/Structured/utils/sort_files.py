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

X, Y = data.samples["VALID"][0], data.samples["VALID"][1]
parser = MarkupParser()

dir = os.path.join(config.train_root_directory, "validation")

for i in range(len(X)):
    if not os.path.exists(dir):
        os.mkdir(dir)

    if not os.path.exists(os.path.join(dir, os.path.basename(X[i]))):
        os.rename(X[i], os.path.join(dir, os.path.basename(X[i])))
        os.rename(Y[i], os.path.join(dir, os.path.basename(Y[i])))

