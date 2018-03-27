from Structured.utils.config import *

import numpy as np


class JSONParser:
    def __init__(self):
        pass


    def getBoundBoxes(self, json_file):
        config, config_dict = get_config_from_json(json_file)
        boxes = config.objects

        bboxes_list = []
        for box in boxes:
            points = np.array(box["points"]["exterior"])

            x, y = np.split(points, 2, axis=1)

            xmin = np.min(x)
            ymin = np.min(y)
            xmax = np.max(x)
            ymax = np.max(y)

            bboxes_list.append(np.stack((xmin, ymin, xmax, ymax)))
        return bboxes_list


if __name__ == '__main__':
    json = JSONParser()

    bboxes = json.getBoundBoxes(
        "C:\\Users\\admin\\Documents\\GeneralProjectData\\Samples\\Licence plates__artificial\\ann\\00000000_E112YB13.json")
    pass
