import sys

import PIL

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import random

from Net.GUI.Demo.mainwindow import Ui_MainWindow as MainWindow


class MainForm(QMainWindow, MainWindow):
    def __init__(self, parent=None):
        super(MainForm, self).__init__(parent)
        self.setupUi(self)

        '''btn_Quit.OnClick -> Close'''
        self.btn_Quit.clicked.connect(self.OnCloseWindowClick)

        '''btn_OpenFile -> Open File Dlg'''
        self.btn_OpenFile.clicked.connect(self.OnLoadWithFileDialogClick)

        self.btn_FrameRecognize.clicked.connect(self.OnRecognizeClick)

        '''Create instance to store loaded img'''
        self.imageContainer = Frame()

        '''Set img preferred size'''
        self.init_label()
        pass

    def init_label(self):
        self.label_Image.setGeometry(0, 0, 800, 452)
        pixm = QPixmap(self.label_Image.size())
        pixm = pixm.scaled(self.label_Image.size(), Qt.KeepAspectRatio)
        self.label_Image.setPixmap(pixm)
        pass

    def update_label_Image(self, pixmap):
        self.label_Image.setPixmap(pixmap)

    # todo doesnt work, pass stub
    '''Закрыть инстанс GUI'''

    def OnCloseWindowClick(self):
        # QCoreApplication.Core.instance().quit()
        pass

    '''Загрузить изображение'''

    def OnLoadWithFileDialogClick(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
            file_info = QFileInfo(file_name)

            if file_info.exists():
                # How to get original name and full name
                # originalName = QFileInfo.completeBaseName(file_info)
                # QFileInfo.baseName()

                path = file_info.absoluteFilePath()
                originalExtenstion = QFileInfo.completeSuffix(file_info)

                self.imageContainer.set_picture(path)
                self.imageContainer.originalExtension = originalExtenstion

                self.imageContainer.dataCopy = self.imageContainer.dataCopy.scaled(self.label_Image.size(),
                                                                                   Qt.KeepAspectRatio)
                self.update_label_Image(self.imageContainer.dataCopy)

                '''Uncomment to draw example on img loaded'''
                # painter  = Painter(self.imageContainer.dataCopy)
                #
                # rnd = random.randint(0, 400)
                # list  = []
                #
                # for i in range(1, 1):
                #     x1 = random.randint(0, 200)
                #     y1 = random.randint(0, 200)
                #     width = random.randint(0, 400)
                #     height = random.randint(0, 400)
                #     rect = QRect(x1, y1, width, height)
                #     list.append(rect)
                #
                # x1 = 200
                # y1 = 200
                # width = 50
                # height = 50
                # rect = QRect(x1, y1, width, height)
                #
                # list.append(rect)
                # # painter.paint_rectangles(list)
                # # painter.paint_rect(rect)
                # # painter.paint_rectangle(x1, y1, width=width, height=height)
                #
                #
                # b = painter.paint_rectangle(100, 100, -120, 120,  None)
                #
                #
                # if b == False:
                #     # dlg = WarnDialog()
                #     # dlg.show()
                #
                #     self.showWarn("Warning", "Одна из областей не была нарисована")
                #     pass
                #
                # self.update_label_Image(self.imageContainer.dataCopy)
                '''block ends'''

                return path

            else:
                return None

        except IOError:
            print("Ошибка загрузки файла")
            return
        pass

    '''Stub method'''

    def OnRecognizeClick(self):
        pass

    def showWarn(parent, title: str, msg: str):
        QMessageBox.about(parent, title, msg)


class Frame():
    def __init__(self, parent=None):
        '''Placeholder to keep image loaded further'''
        self.originalExtension = None

        self.data = QPixmap()

        '''Contains a copy of data at some moment'''
        self.dataCopy = QPixmap()

    '''Construct internal pixmap from QPixmap'''

    def set_picture(self, QPixmap):
        self.data = QPixmap(QPixmap)
        self.dataCopy = self.data.copy()

    '''Construct internal pixmap from path str'''

    def set_picture(self, FullPathToImg):
        self.data = QPixmap(FullPathToImg)
        self.dataCopy = self.data.copy()

    # todo doesn't work =(
    '''Save unmodified data img'''

    def save_image(self, fileFullPath=(QDir.currentPath() + "\\"),
                   fileName="default",
                   extension="jpg"):
        b = self.originalExtension  # example "png"
        pass

        path = fileFullPath + fileName + "." + extension
        self.data.save(path)


class Painter():
    def __init__(self, whereToDraw=None, penColour=None, penSize=None):
        self.painter = QPainter(whereToDraw)
        self.pen = QPen(Qt.red, 3)
        self.painter.setPen(self.pen)

    '''Paint line from (x1;x2) to (y1;y2)'''

    def paint_line(self, x1, y1, x2, y2) -> bool:
        self.painter.drawLine(x1, y1, x2, y2)
        return True
        pass

    '''Paint rectangle (x1; x1) (width; height)'''

    def paint_rectangle(self, x1: int, y1: int, width: int, height: int) -> bool:
        self.painter.drawRect(x1, y1, width, height)
        return True
        pass

    '''Paint rectangle (x1; y1) (x2; y2)'''

    def paint_rectangle(self, x1, y1, x2, y2, _=None) -> bool:
        if (_ is None):
            if (x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
                if (x2 > x1 and y2 > y1):
                    rect = QRect(x1, y1, x2 - x1, y2 - y1)
                    self.painter.drawRect(rect)
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    '''Paint QRect'''

    def paint_rect(self, rect: QRect) -> bool:
        self.painter.drawRect(rect)
        return True
        pass

    '''Paint a list of QRect'''

    def paint_rectangles(self, list: []):
        if (list is not None):
            for rect in list:
                self.paint_rect(rect)
        pass


''' "C:\\Users\\Username" '''


def getHomePath():
    return QDir.homePath()


''' The place where this .py exists'''


def getCurrentPath():
    return QDir.currentPath()


def main():
    application = QApplication(sys.argv)
    mainForm = MainForm()

    mainForm.show()
    application.exec()


if __name__ == '__main__':
    main()
