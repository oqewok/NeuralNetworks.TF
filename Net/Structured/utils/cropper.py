from Structured.data_loader.art_car_plates_data_provider import ArtificalCarPlatesDataProvider
from Structured.data_loader.reader import Reader
from Structured.utils.config import process_config
from skimage import io

import numpy as np
import os

def crop_good():
    config = process_config(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\crop.json")

    data = ArtificalCarPlatesDataProvider(config)

    X, Y = data.samples["TRAIN"][0], data.samples["TRAIN"][1]

    path = "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\cascade\\Good.dat"

    open(path, 'a').close()

    widths = []
    heights = []

    with open(path, "w") as writer:
        for i in range(len(X)):
            base_name = os.path.basename(X[i])
            H, W, C = io.imread(X[i]).shape
            boxes = Reader.parse_bbox_file(Y[i])

            xmin, ymin, xmax, ymax, _ = np.split(boxes[0], 5)
            w = xmax - xmin
            h = ymax - ymin

            widths.append(w / W * 224)
            heights.append(h / H * 224)

            result_str = "Good\\" + base_name + " 1 " + str(xmin[0]) + " " + str(ymin[0]) + " " + str(w[0]) + " " + str(h[0]) + "\n"
            writer.write(result_str)

    ratios = np.array(widths) / np.array(heights)
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


def crop_bad():
    config = process_config(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Projects\\NeuralNetworks.TF\\Net\Structured\\configs\\crop.json")

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
    crop_bad()

