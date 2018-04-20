from Structured.data_loader.car_plates_data_provider import CarPlatesDataProvider
from Structured.data_loader.reader import Reader
from Structured.utils.config import process_config
from skimage import io
import matplotlib.pyplot as plt

import collections
import numpy as np
import os
from tqdm import tqdm

def crop_good():
    config = process_config(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\fastercnn.json")

    data = CarPlatesDataProvider(config)

    X, Y = data.samples["TRAIN"][0], data.samples["TRAIN"][1]

    path = "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\cascade\\Good.dat"

    open(path, 'a').close()

    widths = []
    heights = []

    with open(path, "w") as writer:
        for i in tqdm(range(len(X))):
            newX = X[i]#.replace(" ", "_")
            newY = Y[i]#.replace(" ", "_")

            # os.rename(X[i], newX)
            # os.rename(Y[i], newY)

            base_name = os.path.basename(newX)
            img = io.imread(newX)
            shape = img.shape

            H, W, C = None, None, None
            if len(shape) == 2:
                H, W = shape
            elif len(shape) == 3:
                H, W, C = shape
            else:
                H, W, C = img[0].shape

            try:
                boxes = Reader.parse_bbox_file(newY)

                xmin, ymin, xmax, ymax, _ = np.split(boxes, 5, axis=1)

                r, c = xmin.shape

                # if r > 1:
                #     print()

                w = np.reshape(xmax - xmin, [r])
                h = np.reshape(ymax - ymin, [r])

                for j in range(r):
                    widths.append(w[j] / W * 224)
                    heights.append(h[j] / H * 224)

            except TypeError:
                print(newY)

            # result_str = "Good\\" + base_name + " 1 " + str(xmin[0]) + " " + str(ymin[0]) + " " + str(w[0]) + " " + str(h[0]) + "\n"
            # writer.write(result_str)

    ratios = np.array(widths) / np.array(heights)
    ratios = ratios.reshape(len(ratios))

    c = collections.Counter(ratios)
    k = list(c.keys())
    v = list(c.values())

    print('min ratio', np.min(ratios))
    print('mean ratio', np.mean(ratios))
    print('max ratio', np.max(ratios))
    print('std ratio', np.std(ratios))
    print()
    print('min width', np.min(widths))
    print('meann width', np.mean(widths))
    print('max width', np.max(widths))
    print('std width', np.std(widths))
    print()
    print('min height', np.min(heights))
    print('mean height', np.mean(heights))
    print('max height', np.max(heights))
    print('std height', np.std(heights))

    plt.title("Lol")
    plt.xlabel("ratios")
    plt.ylabel("count")

    #plt.axis([0, np.max(k), 0, np.max(v)])
    plt.axis("auto")
    plt.scatter(k, v, edgecolors='r', s=5)

    plt.grid(True, linestyle=':', color='0.75')

    plt.show()

def crop_bad():
    config = process_config(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\fastercnn.json")

    path = "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\cascade\\Bad.dat"

    open(path, 'a').close()

    with open(path, "w") as writer:
        for root, dirs, files in os.walk("C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\cascade\\Bad\\"):
            files = filter(
                lambda x: x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png') or x.endswith('.bmp'),
                files)
            for file in files:

                result_str = "Bad\\" + file + "\n"
                writer.write(result_str)


if __name__ == '__main__':
    crop_good()
    #crop_bad()

