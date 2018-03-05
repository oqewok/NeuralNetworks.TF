import xml.etree.cElementTree as ET

class MarkupParser():
    def __init__(self):
        pass

    def getHumanCheckedValueAttr(self, xml_file):
        # E:/YandexDisk/testsamples/frames/Беларусь(BY)/2017-09-08 17-20-44.xml
        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        hc = root.find('HumanChecked')
        val = hc.attrib["Value"]

        return bool(val)


'''
p = MarkupParser()
a = p.getHumanCheckedValueAttr("E:/YandexDisk/testsamples/frames/Беларусь(BY)/2017-09-08 17-20-44.xml")
pass
'''