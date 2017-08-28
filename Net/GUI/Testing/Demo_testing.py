import sys
import os
from PyQt5 import QtCore

from PyQt5.QtCore import QCoreApplication as Core, QFileInfo, QUrl, QDir, QSize
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QWidget, QSizePolicy)
from PyQt5.QtMultimedia import QMultimedia as Multimedia, QMediaPlayer, QMediaContent
from PyQt5.QtMultimedia import QMediaRecorder as MediaRecorder
from PIL import Image
from Net.GUI.Testing.mainwindow import Ui_MainWindow as MainWindow


# player = QMediaPlayer
class MainGUIWindow(QMainWindow, MainWindow):
    def __init__(self, parent = None):
        super(MainGUIWindow, self).__init__(parent)
        self.setupUi(self)

        '''btn_Quit.OnClick -> Close'''
        self.btn_Quit.clicked.connect(self.close_window)

        '''btn_OpenFile.OnClick -> Open File Dialog'''
        self.btn_OpenFile.clicked.connect(self.openFile)

        '''btn_FrameShowNext -> Pick and show next frame'''
        self.btn_FrameShowNext.clicked.connect(self.get_next_frame)

        '''btn_FrameShowNext -> Pick and show next frame'''
        self.btn_FrameShowPrevious.clicked.connect(self.get_prev_frame)

        '''Set widgetVideo to display content from picked video file'''
        self.widgetVideo = VideoWidget(parent=self)
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.widgetVideo)

    '''Заглушка'''
    def get_next_frame(self):
       pass

    '''Заглушка'''
    def get_prev_frame(self):
       pass

    '''Воспроизвести файл средствами OS'''
    def open_file_with_os(self):
        try:
            file_name = self.get_file_path()

            if file_name is not None:
                os.startfile(file_name)

        except IOError:
            print("Ошибка загрузки файла")
            sys.exit(1)

    '''Открыть FileDialog, вернуть полное имя файла'''
    def get_file_path(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
            file_info = QFileInfo(file_name)

            if file_info.exists():
                path = file_info.absoluteFilePath()
                return path

            else:
                return None

        except IOError:
            print("Ошибка загрузки файла")
            return

    '''Загрузить видео в виждет'''
    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())

        if fileName != '':
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.player.play()

    '''Закрыть инстанс GUI'''
    def close_window(self):
        Core.instance().quit()

class VideoWidget(QVideoWidget):
    def __init__(self, parent = None):
        super(VideoWidget, self).__init__(parent)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)

        '''Заливка widgetVideo'''
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setPalette(p)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        self.resize(self.sizeHint())

    def sizeHint(self):
        return QSize(450, 300)

# class Frame(QVideoFrame):
#     def __init__(self, parent=None):
#         super(Frame, self).__init__(parent)
#         pass

def main():
    application = QApplication(sys.argv)
    GUI_instance = MainGUIWindow()

    GUI_instance.show()
    application.exec()


if __name__ == '__main__':
    main()
