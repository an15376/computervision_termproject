# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Camera feed QLabel
        self.CameraFeed = QtWidgets.QLabel(self.centralwidget)
        self.CameraFeed.setGeometry(QtCore.QRect(50, 50, 800, 600))
        self.CameraFeed.setObjectName("CameraFeed")
        self.CameraFeed.setStyleSheet("border: 1px solid black;")
        self.CameraFeed.setAlignment(QtCore.Qt.AlignCenter)
        
        # Start/Stop detection QPushButton
        self.DetectButton = QtWidgets.QPushButton(self.centralwidget)
        self.DetectButton.setGeometry(QtCore.QRect(900, 50, 200, 50))
        self.DetectButton.setObjectName("DetectButton")
        
        # Result QLabel
        self.ResultText = QtWidgets.QLabel(self.centralwidget)
        self.ResultText.setGeometry(QtCore.QRect(900, 150, 200, 50))
        self.ResultText.setObjectName("ResultText")
        self.ResultText.setStyleSheet("border: 1px solid black;")
        self.ResultText.setAlignment(QtCore.Qt.AlignCenter)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Camera Object Detection"))
        self.CameraFeed.setText(_translate("MainWindow", "Camera Feed"))
        self.DetectButton.setText(_translate("MainWindow", "Start Object Detection"))
        self.ResultText.setText(_translate("MainWindow", "Result"))
