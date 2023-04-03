# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Farmakod.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Farmakod(object):
    def setupUi(self, Farmakod):
        if not Farmakod.objectName():
            Farmakod.setObjectName(u"Farmakod")
        Farmakod.resize(842, 600)
        self.centralwidget = QWidget(Farmakod)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.buttonBox = QDialogButtonBox(self.centralwidget)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 2, 2, 1, 1)

        self.openGLWidget = QOpenGLWidget(self.centralwidget)
        self.openGLWidget.setObjectName(u"openGLWidget")

        self.gridLayout.addWidget(self.openGLWidget, 1, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setSpacing(12)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.checkBox = QCheckBox(self.centralwidget)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setTabletTracking(False)
        self.checkBox.setChecked(True)

        self.verticalLayout_4.addWidget(self.checkBox)

        self.checkBox_2 = QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setChecked(True)

        self.verticalLayout_4.addWidget(self.checkBox_2)

        self.checkBox_3 = QCheckBox(self.centralwidget)
        self.checkBox_3.setObjectName(u"checkBox_3")
        self.checkBox_3.setChecked(True)

        self.verticalLayout_4.addWidget(self.checkBox_3)

        self.checkBox_4 = QCheckBox(self.centralwidget)
        self.checkBox_4.setObjectName(u"checkBox_4")
        self.checkBox_4.setChecked(True)

        self.verticalLayout_4.addWidget(self.checkBox_4)


        self.horizontalLayout_4.addLayout(self.verticalLayout_4)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setMouseTracking(True)
        self.horizontalSlider.setTabletTracking(True)
        self.horizontalSlider.setAutoFillBackground(False)
        self.horizontalSlider.setValue(10)
        self.horizontalSlider.setOrientation(Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)

        self.verticalLayout_3.addWidget(self.horizontalSlider)

        self.horizontalSlider_2 = QSlider(self.centralwidget)
        self.horizontalSlider_2.setObjectName(u"horizontalSlider_2")
        self.horizontalSlider_2.setValue(90)
        self.horizontalSlider_2.setOrientation(Qt.Horizontal)

        self.verticalLayout_3.addWidget(self.horizontalSlider_2)

        self.horizontalSlider_3 = QSlider(self.centralwidget)
        self.horizontalSlider_3.setObjectName(u"horizontalSlider_3")
        self.horizontalSlider_3.setValue(10)
        self.horizontalSlider_3.setOrientation(Qt.Horizontal)

        self.verticalLayout_3.addWidget(self.horizontalSlider_3)

        self.horizontalSlider_4 = QSlider(self.centralwidget)
        self.horizontalSlider_4.setObjectName(u"horizontalSlider_4")
        self.horizontalSlider_4.setValue(99)
        self.horizontalSlider_4.setOrientation(Qt.Horizontal)

        self.verticalLayout_3.addWidget(self.horizontalSlider_4)


        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lcdNumber = QLCDNumber(self.centralwidget)
        self.lcdNumber.setObjectName(u"lcdNumber")
        self.lcdNumber.setProperty("value", 10.000000000000000)

        self.verticalLayout_2.addWidget(self.lcdNumber)

        self.lcdNumber_2 = QLCDNumber(self.centralwidget)
        self.lcdNumber_2.setObjectName(u"lcdNumber_2")
        self.lcdNumber_2.setProperty("value", 90.000000000000000)

        self.verticalLayout_2.addWidget(self.lcdNumber_2)

        self.lcdNumber_3 = QLCDNumber(self.centralwidget)
        self.lcdNumber_3.setObjectName(u"lcdNumber_3")
        self.lcdNumber_3.setProperty("value", 10.000000000000000)

        self.verticalLayout_2.addWidget(self.lcdNumber_3)

        self.lcdNumber_4 = QLCDNumber(self.centralwidget)
        self.lcdNumber_4.setObjectName(u"lcdNumber_4")
        self.lcdNumber_4.setProperty("value", 99.000000000000000)

        self.verticalLayout_2.addWidget(self.lcdNumber_4)


        self.horizontalLayout_4.addLayout(self.verticalLayout_2)


        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 3)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Plain)
        self.frame.setLineWidth(10)
        self.line = QFrame(self.frame)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(20, 0, 41, 251))
        self.line.setLineWidth(3)
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line_2 = QFrame(self.frame)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(330, 0, 31, 261))
        self.line_2.setLineWidth(3)
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.line_3 = QFrame(self.frame)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(0, 20, 381, 20))
        self.line_3.setLineWidth(3)
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)
        self.line_4 = QFrame(self.frame)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setGeometry(QRect(-30, 220, 411, 31))
        self.line_4.setLineWidth(3)
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(120, 90, 131, 71))

        self.gridLayout.addWidget(self.frame, 1, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        Farmakod.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Farmakod)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 842, 20))
        self.menuAyarlar = QMenu(self.menubar)
        self.menuAyarlar.setObjectName(u"menuAyarlar")
        self.menuDosya = QMenu(self.menubar)
        self.menuDosya.setObjectName(u"menuDosya")
        Farmakod.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuDosya.menuAction())
        self.menubar.addAction(self.menuAyarlar.menuAction())

        self.retranslateUi(Farmakod)
        self.horizontalSlider.valueChanged.connect(self.lcdNumber.display)
        self.horizontalSlider_2.valueChanged.connect(self.lcdNumber_2.display)
        self.horizontalSlider_3.valueChanged.connect(self.lcdNumber_3.display)
        self.horizontalSlider_4.valueChanged.connect(self.lcdNumber_4.display)
        self.buttonBox.clicked.connect(self.buttonBox.update)
        self.checkBox.toggled.connect(self.line.setVisible)
        self.checkBox_2.toggled.connect(self.line_2.setVisible)
        self.checkBox_3.toggled.connect(self.line_3.setVisible)
        self.checkBox_4.toggled.connect(self.line_4.setVisible)

        QMetaObject.connectSlotsByName(Farmakod)
    # setupUi

    def retranslateUi(self, Farmakod):
        Farmakod.setWindowTitle(QCoreApplication.translate("Farmakod", u"Farmakod Ayarlar ", None))
        self.checkBox.setText(QCoreApplication.translate("Farmakod", u"Tehlikeli b\u00f6lge SOL s\u0131n\u0131r\u0131 % olarak", None))
        self.checkBox_2.setText(QCoreApplication.translate("Farmakod", u"Tehlikeli b\u00f6lge SA\u011e s\u0131n\u0131r\u0131 % olarak", None))
        self.checkBox_3.setText(QCoreApplication.translate("Farmakod", u"Tehlikeli b\u00f6lge \u00dcST s\u0131n\u0131r\u0131 % olarak", None))
        self.checkBox_4.setText(QCoreApplication.translate("Farmakod", u"Tehlikeli b\u00f6lge ALT s\u0131n\u0131r\u0131 % olarak", None))
#if QT_CONFIG(tooltip)
        self.horizontalSlider.setToolTip(QCoreApplication.translate("Farmakod", u"% olarak \u00fcst b\u00f6lgeyi ayarlay\u0131n\u0131z", None))
#endif // QT_CONFIG(tooltip)
        self.label_2.setText(QCoreApplication.translate("Farmakod", u"Tehlikeli B\u00f6lge Alan\u0131", None))
        self.menuAyarlar.setTitle(QCoreApplication.translate("Farmakod", u"Ayarlar", None))
        self.menuDosya.setTitle(QCoreApplication.translate("Farmakod", u"Dosya", None))
    # retranslateUi

