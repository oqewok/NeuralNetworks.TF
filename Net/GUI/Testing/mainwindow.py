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
        MainWindow.resize(700, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_OpenClose = QtWidgets.QVBoxLayout()
        self.verticalLayout_OpenClose.setObjectName("verticalLayout_OpenClose")
        self.btn_OpenFile = QtWidgets.QPushButton(self.centralwidget)
        self.btn_OpenFile.setObjectName("btn_OpenFile")
        self.verticalLayout_OpenClose.addWidget(self.btn_OpenFile)
        self.btn_Quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Quit.setObjectName("btn_Quit")
        self.verticalLayout_OpenClose.addWidget(self.btn_Quit)
        self.gridLayout.addLayout(self.verticalLayout_OpenClose, 0, 3, 1, 1)
        self.horizontalLayout_btn_PrevNext = QtWidgets.QHBoxLayout()
        self.horizontalLayout_btn_PrevNext.setSpacing(6)
        self.horizontalLayout_btn_PrevNext.setObjectName("horizontalLayout_btn_PrevNext")
        self.btn_FrameShowPrevious = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameShowPrevious.setObjectName("btn_FrameShowPrevious")
        self.horizontalLayout_btn_PrevNext.addWidget(self.btn_FrameShowPrevious)
        self.btn_FrameRecognize = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameRecognize.setObjectName("btn_FrameRecognize")
        self.horizontalLayout_btn_PrevNext.addWidget(self.btn_FrameRecognize)
        self.btn_FrameShowNext = QtWidgets.QPushButton(self.centralwidget)
        self.btn_FrameShowNext.setObjectName("btn_FrameShowNext")
        self.horizontalLayout_btn_PrevNext.addWidget(self.btn_FrameShowNext)
        self.gridLayout.addLayout(self.horizontalLayout_btn_PrevNext, 4, 0, 1, 3)
        self.verticalLayout_widgetVideo = QtWidgets.QVBoxLayout()
        self.verticalLayout_widgetVideo.setObjectName("verticalLayout_widgetVideo")
        self.widgetVideo = QtWidgets.QWidget(self.centralwidget)
        self.widgetVideo.setMaximumSize(QtCore.QSize(508, 299))
        self.widgetVideo.setObjectName("widgetVideo")
        self.verticalLayout_widgetVideo.addWidget(self.widgetVideo)
        self.gridLayout.addLayout(self.verticalLayout_widgetVideo, 0, 0, 4, 3)
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
        self.btn_OpenFile.setText(_translate("MainWindow", "Открыть"))
        self.btn_Quit.setText(_translate("MainWindow", "Выйти"))
        self.btn_FrameShowPrevious.setText(_translate("MainWindow", "<<"))
        self.btn_FrameRecognize.setText(_translate("MainWindow", "Обнаружить"))
        self.btn_FrameShowNext.setText(_translate("MainWindow", ">>"))
        self.action.setText(_translate("MainWindow", "Открыть"))
        self.action_3.setText(_translate("MainWindow", "Выход"))

