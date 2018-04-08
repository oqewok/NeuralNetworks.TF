import xml.etree.cElementTree as ET
import xml
import numpy as np

COUNTRIES = {
    "abh" : 0,
    "at"  : 1,
    "by"  : 2,
    "be"  : 3,
    "de"  : 4,
    "lt"  : 5,
    "ru"  : 6,
    "uz"  : 7,
    "ua"  : 8,
    "kz"  : 9,
    "lv"  : 10,
    "fi"  : 11,
}

COUNTRIES_R = {
    0   : "abh",
    1   : "at",
    2   : "by",
    3   : "be",
    4   : "de",
    5   : "lt",
    6   : "ru",
    7   : "uz",
    8   : "ua",
    9   : "kz",
    10  : "lv",
    11  : "fi",
}

class MarkupParser:
    def __init__(self):
        pass

    def getHumanCheckedValueAttr(self, xml_file):
        try:
            tree = ET.ElementTree(file=xml_file)
            root = tree.getroot()
            hc = root.find('HumanChecked')
            val = hc.attrib["Value"]

            if val == "True" or val == "true":
                return True
            else:
                return False
        except xml.etree.ElementTree.ParseError:
            print(xml_file)




    def getCountry(self, xml_file):
        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        hc = root.find('Description')
        val = hc.attrib["Value"]

        if val == "":
            print(xml_file)

        return val



    def getBoundBoxes(self, xml_file):
        plates = []

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        for plate in root.iter("Plate"):
            x, y = [], []
            country = plate.find("Country")
            c = country.attrib["Value"]

            for region in plate.iter("Region"):
                for point in region.iter("Point"):
                    x.append(int(point.attrib["X"]))
                    y.append(int(point.attrib["Y"]))

                '''xcenter = abs(int(
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
                '''
                xmin = np.min(x)
                ymin = np.min(y)
                xmax = np.max(x)
                ymax = np.max(y)

                label = None
                try:
                    label = COUNTRIES[c]
                except KeyError:
                    print(xml_file)

                plates.append(np.stack((xmin, ymin, xmax, ymax, label)))

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

