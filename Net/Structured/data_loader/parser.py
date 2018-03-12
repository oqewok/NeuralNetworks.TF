import xml.etree.cElementTree as ET
import numpy as np


class MarkupParser:
    def __init__(self):
        pass

    def getHumanCheckedValueAttr(self, xml_file):
        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        hc = root.find('HumanChecked')
        val = hc.attrib["Value"]

        if val == "True" or val == "true": return True
        else: return False

    def getBoundBoxes(self, xml_file):
        plates = []

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        for plate in root.iter("Plate"):
            x, y = [], []
            for region in plate.iter("Region"):
                for point in region.iter("Point"):
                    x.append(int(point.attrib["X"]))
                    y.append(int(point.attrib["Y"]))

                xcenter = abs(int(
                    0.5 * (max(x) + min(x))
                ))
                ycenter = abs(int(
                    0.5 * (max(y) + min(y))
                ))
                width   = abs(int(
                    max(x) - min(x)
                ))
                height  = abs(int(
                    max(y) - min(y)
                ))

                plates.append(np.stack((xcenter, ycenter, width, height)))

        plates = np.array(plates, dtype=int)

        return plates

"""
p = MarkupParser()
a = p.getBoundBoxes("E:/YandexDisk/testsamples/frames/Россия(RU)/2017-06-17 17-25-33.xml")
pass
"""



'''
p = MarkupParser()
a = p.getHumanCheckedValueAttr("E:/YandexDisk/testsamples/frames/Абхазия(AB)/ABH (avto-nomer.ru)/car10387298.xml")
pass
'''

