from Structured.utils.config import process_config
from Structured.data_loader.car_plates_data_provider import CarPlatesDataProvider
from tqdm import tqdm

import numpy as np

config = process_config("C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\\Structured\\configs\\fastercnn.json")
data = CarPlatesDataProvider(config)

r = 0
g = 0
b = 0

loop = tqdm(range(data.num_train))
for i in loop:
    img, _ = data.next_img()

    r += np.mean(img[:, :, 0])
    g += np.mean(img[:, :, 1])
    b += np.mean(img[:, :, 2])

r /= data.num_train
g /= data.num_train
b /= data.num_train

print(r)
print(g)
print(b)
