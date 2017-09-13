import sys

import PIL

import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import random
import numpy as np


import tensorflow as tf

from mainwindow import Ui_MainWindow as MainWindow


class MainForm(QMainWindow, MainWindow):
    def __init__(self, parent=None):
        super(MainForm, self).__init__(parent)
        self.setupUi(self)

        '''btn_Quit.OnClick -> Close'''
        self.btn_Quit.clicked.connect(self.OnCloseWindowClick)

        '''btn_OpenFile -> Open File Dlg'''
        self.btn_OpenFile.clicked.connect(self.OnLoadImageClick)

        '''btn_FrameRecognize -> Reconize with tf model'''
        self.btn_FrameRecognize.clicked.connect(self.OnRecognizeClick)

        '''btn_LoadModelMeta - > Load Model's .meta'''
        self.btn_LoadModelMeta.clicked.connect(self.OnLoadModelMetaClick)

        '''Create instance to store loaded img'''
        self.imageContainer = Frame()

        '''Set img preferred size'''
        self.init_label()

        '''Makes session instance'''
        self.TFSession = TFSessionHolder()
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

    def OnLoadImageClick(self):
        try:
            file_filter_string = "*.jpg; *.jpeg; *.png; *.bmp;; *.jpg;; *.*"
            file_name, _ = QFileDialog.getOpenFileName(self, caption="Выберите файл", filter=file_filter_string)
            file_info = QFileInfo(file_name)

            if file_info.exists():
                # # How to get original name and full name
                # originalName = QFileInfo.completeBaseName(file_info)
                # p = QFileInfo.baseName(file_info)

                path = file_info.absoluteFilePath()
                originalExtension = QFileInfo.completeSuffix(file_info)

                self.imageContainer.set_picture(path)
                self.imageContainer.originalExtension = originalExtension

                self.imageContainer.dataCopy = self.imageContainer.dataCopy.scaled(self.label_Image.size(),
                                                                                   Qt.KeepAspectRatio)
                self.update_label_Image(self.imageContainer.dataCopy)


                '''Uncomment to draw example on img loaded'''
                painter  = Painter(self.imageContainer.dataCopy)

                rnd = random.randint(0, 400)
                list  = []

                # for i in range(1, 1):
                #     x1 = random.randint(0, 200)
                #     y1 = random.randint(0, 200)
                #     width = random.randint(0, 400)
                #     height = random.randint(0, 400)
                #     rect = QRect(x1, y1, width, height)
                #     list.append(rect)

                x1 = 200
                y1 = 200
                width = 50
                height = 50
                rect = QRect(x1, y1, width, height)

                list.append(rect)
                # painter.paint_rectangles(list)
                # painter.paint_rect(rect)
                # painter.paint_rectangle(x1, y1, width=width, height=height)


                b = painter.paint_rectangle(10, 10, 11, 11,  None)


                if b == False:
                    self.showWarn("Warning", "Одна из областей не была нарисована")
                    pass

                self.update_label_Image(self.imageContainer.dataCopy)
                '''block ends'''


                '''wip'''
                # r, c = self.TFSession.evaluate(None)
                #
                # # coord_arr = np.asarray(c)
                #
                # print(c)
                # painter  = Painter(self.imageContainer.dataCopy)
                # if r is True:
                #     painter.paint_rectangles(c)
                #     painter.paint_rectangle(c)
                #     self.update_label_Image(self.imageContainer.dataCopy)
                '''wip ends'''

                print(self.imageContainer.getOriginalHeigth())
                print(self.imageContainer.getOriginalWidth())


                pass
                return path

            else:
                return None

        except IOError:
            print("Ошибка загрузки файла")
            traceback.format_exc()
            return
        pass

    '''Загрузить .meta и восстановить tf-сессию'''

    def OnLoadModelMetaClick(self):
        try:

            file_filter_string = "*.meta;;*.*"
            file_name, _ = QFileDialog.getOpenFileName(self, caption="Выберите файл модели", directory=getCurrentPath(), filter=file_filter_string)
            file_info = QFileInfo(file_name)

            if file_info.exists():
                fullName = file_info.absoluteFilePath()
                folderName = file_info.path();

            else:
                self.showWarn("Ошибка", "Модель не была загружена, т.к. отсутствует файл")
                return

            with open('log.txt', 'a') as f:
                f.write("\n\n\nLoading model.\n\n")

            result = self.TFSession.load_model(FullFolderNameToModel=folderName, FullMetaFileName=fullName)

            if (result is not True):
                self.showWarn(self.showWarn("Ошибка", "Модель не была загружена. Проверьте лог-файл"))
        except:
            self.showWarn("Ошибка", "Ошибка при загрузке модели")

    '''Evaluate with model loaded previously'''

    def OnRecognizeClick(self):
        try:
            if (self.imageContainer.dataCopy is not None):

                result, coord = self.TFSession.evaluate(self.imageContainer.dataCopy)

                if (result is True):
                    painter = Painter(self.imageContainer.dataCopy)

                    b = painter.paint_rect(coord[0])

                    if b == False:
                        self.showWarn("Warning", "Одна из областей не была нарисована")
                        pass

                    self.update_label_Image(self.imageContainer.dataCopy)

                    pass

                else:
                    self.showWarn("", "Оценка не произведена")

            else:
                self.showWarn("Ошибка", "Изображение не загружено")


        except Exception as e:
            with open('log.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
                traceback.format_exc()

                msg = str(e)
                self.showWarn("Ошибка", msg)

            pass
        pass

    '''Can be called to show an info'''

    def showWarn(parent, title: str, msg: str):
        QMessageBox.about(parent, title, msg)


class Frame():
    def __init__(self, parent=None):
        '''Placeholder to keep image loaded further'''
        self.originalExtension = None

        self.data = QPixmap()

        '''Contains a copy of data at some moment'''
        self.dataCopy = QPixmap()

        self.originalHeight = -1
        self.originalWidth = -1

    '''Construct internal pixmap from QPixmap and set original height and width'''

    def set_picture(self, QPixmap):
        self.data = QPixmap(QPixmap)
        self.dataCopy = self.data.copy()
        self.setInternalOriginalSizes()

    '''Construct internal pixmap from path str'''

    def set_picture(self, FullPathToImg):
        self.data = QPixmap(FullPathToImg)
        self.dataCopy = self.data.copy()
        self.setInternalOriginalSizes()

    def setInternalOriginalSizes(self):
        self.originalHeight = self.data.height()
        self.originalWidth = self.data.width()

    def getOriginalHeigth(self) -> int:
        return self.originalHeight

    def getOriginalWidth(self) -> int:
        return self.originalWidth

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
        self.brushColor = QColor(100, 200, 0)
        self.brushWidth = 2

        self.painter = QPainter(whereToDraw)
        self.pen = QPen(self.brushColor, self.brushWidth)
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
            if (x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0):
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

    '''Paint QRect list'''
    # def paint_rectangle(self, QRectList: [QRect]) -> bool:
    #
    #     return False

    '''Paint QRect'''

    def paint_rect(self, rect: QRect) -> bool:
        try:
            self.painter.drawRect(rect)
            return True

        except Exception:
            traceback.format_exc()
            return False
        pass

    '''Paint a list of QRect'''

    def paint_rectangles(self, list: []):
        if (list is not None):
            for rect in list:
                self.paint_rect(rect)
        pass


class TFSessionHolder():



    def __init__(self):
        self.sess = tf.Session()
        pass

    def load_model(self, FullFolderNameToModel, FullMetaFileName) -> bool:
        try:
            self.saver = tf.train.import_meta_graph(FullMetaFileName)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(FullFolderNameToModel))
            self.ops = tf.get_collection(
                'ops_to_restore')  # here are your operators in the same order in which you saved them to the collection

            self.graph = tf.get_default_graph()
            self.x = self.graph.get_tensor_by_name('inputs:0')
            self.y = self.graph.get_tensor_by_name('outputs:0')

            self.dropout = self.graph.get_tensor_by_name('fc2_c_dropout:0')
            # train_step = graph.get_operation_by_name('Adam')

            self.sess.run(tf.global_variables_initializer())

            print("loaded.")

            return True
            pass

        except Exception as e:
            with open('log.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
                traceback.format_exc()

            return False
        pass


    #todo write evaluation
    def evaluate(self, img):
        try:
            result = True

            coord = []
            rect = QRect(230, 170, 140, 70)
            coord.append(rect)

            return result, coord

        except:
            traceback.format_exc()
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
