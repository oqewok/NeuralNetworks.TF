import sys
import os

import PIL
from PyQt5.QtCore import QCoreApplication, QDir, QUrl, QFileInfo
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtMultimedia import QMediaContent
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from Net.GUI.Demo.mainwindow import Ui_MainWindow as MainWindow

class MainForm(QMainWindow, MainWindow):
    def __init__(self, parent = None):
        super(MainForm, self).__init__(parent)
        self.setupUi(self)

        self.imageFrame = Frame()

        self.btn_OpenFile.clicked.connect(self.load_with_file_dialog)





        self.myPixmap = QPixmap("D:\\FFOutput\\1234m.png")
        self.myPixmap = QPixmap("D:\\output_pic1.jpg")
        self.label_Image.setPixmap(self.myPixmap)


        # Изображение до размера label и показать
        self.label_Image.setGeometry(0, 0, 800, 452)
        self.myPixmap = self.myPixmap.scaled(self.label_Image.size(), Qt.KeepAspectRatio)
        self.label_Image.setPixmap(self.myPixmap)
        # self.label_Image.setScaledContents(True)

        self.label_Image.show()

        self.painter = QPainter(self.myPixmap)

        self.pen = QPen(Qt.red, 0)
        self.painter.setPen(self.pen)
        self.painter.drawLine(self.label_Image.x(), self.label_Image.x(), 400, 400)

        self.label_Image.setPixmap(self.myPixmap)
        pass



    '''Закрыть инстанс GUI'''
    def close_window(self):
        QCoreApplication.Core.instance().quit()


    '''Загрузить изображение'''
    def load_with_file_dialog(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
            file_info = QFileInfo(file_name)

            if file_info.exists():
                path = file_info.absoluteFilePath()
                self.imageFrame.data = QImage(path)

                # self.imageFrame.save("D:\\out.jpg")
                self.imageFrame.save_image(fileFullPath="D:\\output_pic1.jpg")
                self.imageFrame.setPicture()
                self.imageFrame.picture.save("D:\\pic_out.jpg")
                return path

            else:
                return None

        except IOError:
            print("Ошибка загрузки файла")
            return
        pass

    # 1
    #'''Нарисовать линии'''
    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #     self.pixmap = QPixmap("D:\\pic_out.jpg")
    #     painter.drawPixmap(self.rect(), self.pixmap)
    #     pen = QPen(Qt.red, 3)
    #     painter.setPen(pen)
    #     painter.drawLine(10, 10, self.rect().width() - 10, 10)



class Frame(QImage):
    def __init__(self, parent=None):
        super(Frame, self).__init__(parent)

        self.data = None

        self.picture = QPixmap()


    def setPicture(self):
        self.picture = QPixmap(self.data)


    def save_image(self, fileFullPath = "D:\\out.jpg"):
        self.data.save(fileFullPath)


def getHomePath():
    return QDir.homePath()

def main():
    application = QApplication(sys.argv)
    mainForm = MainForm()

    mainForm.show()
    application.exec()


if __name__ == '__main__':
    main()