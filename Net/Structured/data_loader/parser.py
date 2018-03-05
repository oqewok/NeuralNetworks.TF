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

    def getLabels(self, xml_file):
        plates = []

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        for plate in root.iter("Plate"):
            x, y = [], []
            for region in plate.iter("Region"):
                for point in region.iter("Point"):
                    x.append(point.attrib["X"])
                    y.append(point.attrib["Y"])

                x = np.array(x, dtype=int)
                y = np.array(y, dtype=int)

                xmin = np.min(x)
                xmax = np.max(x)
                ymin = np.min(y)
                ymax = np.max(y)

                plates.append([xmin, ymin, xmax, ymax])

        plates = np.array(plates)

        return plates

'''
p = MarkupParser()
a = p.getLabels("E:/YandexDisk/testsamples/frames/Россия(RU)/2017-06-17 17-25-33.xml")
pass
'''


'''
p = MarkupParser()
a = p.getHumanCheckedValueAttr("E:/YandexDisk/testsamples/frames/Абхазия(AB)/ABH (avto-nomer.ru)/car10387298.xml")
pass
'''
