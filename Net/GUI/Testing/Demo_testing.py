import sys

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog)

from Net.GUI.Testing.mainwindow import Ui_MainWindow as MainWindow


class ExampleApp(QMainWindow, MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # btn_Quit.OnClick -> Close
        self.btn_Quit.clicked.connect(self.close_window)

        # btn_Quit.OnClick -> Close
        self.btn_OpenFile.clicked.connect(self.open_file)

    # func to close an instance of gui app
    def close_window(self):
        QCoreApplication.instance().quit()

    # func to open file
    def open_file(self):
        fileName = QFileDialog.getOpenFileName(self, "Выберите файл")


def main():
    app = QApplication(sys.argv)
    win = ExampleApp()
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
