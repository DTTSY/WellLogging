# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DrillingPressureCalculator_window.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSizePolicy,
    QWidget)

class Ui_w_DrillingPressureCalculator(object):
    def setupUi(self, w_DrillingPressureCalculator):
        if not w_DrillingPressureCalculator.objectName():
            w_DrillingPressureCalculator.setObjectName(u"w_DrillingPressureCalculator")
        w_DrillingPressureCalculator.resize(411, 413)
        self.gridLayout = QGridLayout(w_DrillingPressureCalculator)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(w_DrillingPressureCalculator)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.SpanningRole, self.label_4)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.SpanningRole, self.label)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.dsb_Dh = QDoubleSpinBox(self.groupBox)
        self.dsb_Dh.setObjectName(u"dsb_Dh")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.dsb_Dh)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_6)

        self.dsb_Dhi = QDoubleSpinBox(self.groupBox)
        self.dsb_Dhi.setObjectName(u"dsb_Dhi")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.FieldRole, self.dsb_Dhi)

        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.formLayout.setWidget(6, QFormLayout.ItemRole.SpanningRole, self.label_7)

        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")

        self.formLayout.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_8)

        self.dsb_fai300 = QDoubleSpinBox(self.groupBox)
        self.dsb_fai300.setObjectName(u"dsb_fai300")

        self.formLayout.setWidget(7, QFormLayout.ItemRole.FieldRole, self.dsb_fai300)

        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")

        self.formLayout.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_9)

        self.dsb_fai600 = QDoubleSpinBox(self.groupBox)
        self.dsb_fai600.setObjectName(u"dsb_fai600")

        self.formLayout.setWidget(8, QFormLayout.ItemRole.FieldRole, self.dsb_fai600)

        self.pb_open_main = QPushButton(self.groupBox)
        self.pb_open_main.setObjectName(u"pb_open_main")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.pb_open_main)

        self.pb_open_dpf = QPushButton(self.groupBox)
        self.pb_open_dpf.setObjectName(u"pb_open_dpf")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.pb_open_dpf)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 2)

        self.pb_up = QPushButton(w_DrillingPressureCalculator)
        self.pb_up.setObjectName(u"pb_up")

        self.gridLayout.addWidget(self.pb_up, 1, 0, 1, 1)

        self.pb_down = QPushButton(w_DrillingPressureCalculator)
        self.pb_down.setObjectName(u"pb_down")

        self.gridLayout.addWidget(self.pb_down, 1, 1, 1, 1)


        self.retranslateUi(w_DrillingPressureCalculator)

        QMetaObject.connectSlotsByName(w_DrillingPressureCalculator)
    # setupUi

    def retranslateUi(self, w_DrillingPressureCalculator):
        w_DrillingPressureCalculator.setWindowTitle(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u7a33\u5b9a\u6027\u5206\u6790", None))
        self.groupBox.setTitle(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u538b\u529b\u6ce2\u52a8\u52a8\u6001\u8ba1\u7b97", None))
        self.label_4.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u5bfc\u5165\u6570\u636e", None))
        self.label_2.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u5bfc\u5165\u4e3b\u6570\u636e\u6587\u4ef6", None))
        self.label_3.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u574d\u584c\u7834\u88c2\u538b\u529b\u6587\u4ef6", None))
        self.label.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u4e95\u7b52\u53c2\u6570", None))
        self.label_5.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"Dh\u503c (\u94bb\u6746\u5916\u5f84, mm)", None))
        self.label_6.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"Dhi\u503c (\u94bb\u6746\u5185\u5f84, mm)", None))
        self.label_7.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u6d41\u53d8\u53c2\u6570", None))
        self.label_8.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"fai300\u503c (\u516d\u901f\u7c98\u5ea6\u8ba1300\u8f6c\u8bfb\u6570)", None))
        self.label_9.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"fai600\u503c (\u516d\u901f\u7c98\u5ea6\u8ba1600\u8f6c\u8bfb\u6570)", None))
        self.pb_open_main.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u6253\u5f00\u6587\u4ef6", None))
        self.pb_open_dpf.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u6253\u5f00\u6587\u4ef6", None))
        self.pb_up.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u8d77\u94bb", None))
        self.pb_down.setText(QCoreApplication.translate("w_DrillingPressureCalculator", u"\u4e0b\u94bb", None))
    # retranslateUi

