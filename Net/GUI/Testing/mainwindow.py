# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(447, 256)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_FrameShowNext = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameShowNext.setObjectName("btn_FrameShowNext")
        self.gridLayout.addWidget(self.btn_FrameShowNext, 5, 2, 1, 1)
        self.btn_FrameRecognize = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameRecognize.setObjectName("btn_FrameRecognize")
        self.gridLayout.addWidget(self.btn_FrameRecognize, 6, 1, 1, 1)
        self.btn_FrameShowPrevious = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameShowPrevious.setObjectName("btn_FrameShowPrevious")
        self.gridLayout.addWidget(self.btn_FrameShowPrevious, 5, 0, 1, 1)
        self.btn_OpenFile = QtWidgets.QPushButton(self.centralwidget)
        self.btn_OpenFile.setObjectName("btn_OpenFile")
        self.gridLayout.addWidget(self.btn_OpenFile, 0, 3, 1, 1)
        self.btn_Quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Quit.setObjectName("btn_Quit")
        self.gridLayout.addWidget(self.btn_Quit, 1, 3, 1, 1)
        self.graphicsViewVideo = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsViewVideo.setObjectName("graphicsViewVideo")
        self.gridLayout.addWidget(self.graphicsViewVideo, 0, 0, 4, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo"))
        self.btn_FrameShowNext.setText(_translate("MainWindow", ">>"))
        self.btn_FrameRecognize.setText(_translate("MainWindow", "Обнаружить"))
        self.btn_FrameShowPrevious.setText(_translate("MainWindow", "<<"))
        self.btn_OpenFile.setText(_translate("MainWindow", "Открыть"))
        self.btn_Quit.setText(_translate("MainWindow", "Выйти"))
        self.action.setText(_translate("MainWindow", "Открыть"))
        self.action_3.setText(_translate("MainWindow", "Выход"))

