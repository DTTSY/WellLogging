# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QToolBox, QVBoxLayout, QWidget)

class Ui_APPMainWindow(object):
    def setupUi(self, APPMainWindow):
        if not APPMainWindow.objectName():
            APPMainWindow.setObjectName(u"APPMainWindow")
        APPMainWindow.resize(731, 626)
        self.actiondsad = QAction(APPMainWindow)
        self.actiondsad.setObjectName(u"actiondsad")
        self.actiondasd = QAction(APPMainWindow)
        self.actiondasd.setObjectName(u"actiondasd")
        self.actiondasd_2 = QAction(APPMainWindow)
        self.actiondasd_2.setObjectName(u"actiondasd_2")
        self.action_import_Ddata = QAction(APPMainWindow)
        self.action_import_Ddata.setObjectName(u"action_import_Ddata")
        self.action_import_DTdata = QAction(APPMainWindow)
        self.action_import_DTdata.setObjectName(u"action_import_DTdata")
        self.action_subp1 = QAction(APPMainWindow)
        self.action_subp1.setObjectName(u"action_subp1")
        self.action_subp2 = QAction(APPMainWindow)
        self.action_subp2.setObjectName(u"action_subp2")
        self.action_subp3 = QAction(APPMainWindow)
        self.action_subp3.setObjectName(u"action_subp3")
        self.action_conf = QAction(APPMainWindow)
        self.action_conf.setObjectName(u"action_conf")
        self.action_helpDocs = QAction(APPMainWindow)
        self.action_helpDocs.setObjectName(u"action_helpDocs")
        self.centralwidget = QWidget(APPMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(300, 400))
        self.groupBox.setMaximumSize(QSize(340, 16777215))
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(4, 4, 4, 4)
        self.toolBox = QToolBox(self.groupBox)
        self.toolBox.setObjectName(u"toolBox")
        self.toolBox.setMaximumSize(QSize(16777215, 16777215))
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.page.setGeometry(QRect(0, 0, 330, 487))
        self.formLayout = QFormLayout(self.page)
        self.formLayout.setObjectName(u"formLayout")
        self.groupBox_3 = QGroupBox(self.page)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMaximumSize(QSize(16777215, 16777215))
        self.formLayout_2 = QFormLayout(self.groupBox_3)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.dsp_startDepth = QDoubleSpinBox(self.groupBox_3)
        self.dsp_startDepth.setObjectName(u"dsp_startDepth")
        self.dsp_startDepth.setDecimals(3)
        self.dsp_startDepth.setMaximum(999999.000000000000000)

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.dsp_startDepth)

        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.dsp_endDepth = QDoubleSpinBox(self.groupBox_3)
        self.dsp_endDepth.setObjectName(u"dsp_endDepth")
        self.dsp_endDepth.setDecimals(3)
        self.dsp_endDepth.setMaximum(999999.000000000000000)

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.FieldRole, self.dsp_endDepth)

        self.pb_adjDepth = QPushButton(self.groupBox_3)
        self.pb_adjDepth.setObjectName(u"pb_adjDepth")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.SpanningRole, self.pb_adjDepth)


        self.formLayout.setWidget(0, QFormLayout.ItemRole.SpanningRole, self.groupBox_3)

        self.toolBox.addItem(self.page, u"\u6df1\u5ea6\u8c03\u6574")
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.page_2.setGeometry(QRect(0, 0, 330, 487))
        self.gridLayout = QGridLayout(self.page_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox_4 = QGroupBox(self.page_2)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.pb_homorock = QPushButton(self.groupBox_4)
        self.pb_homorock.setObjectName(u"pb_homorock")

        self.verticalLayout_2.addWidget(self.pb_homorock)

        self.pb_added_homorock = QPushButton(self.groupBox_4)
        self.pb_added_homorock.setObjectName(u"pb_added_homorock")

        self.verticalLayout_2.addWidget(self.pb_added_homorock)

        self.pb_drilling_velocity = QPushButton(self.groupBox_4)
        self.pb_drilling_velocity.setObjectName(u"pb_drilling_velocity")

        self.verticalLayout_2.addWidget(self.pb_drilling_velocity)


        self.gridLayout.addWidget(self.groupBox_4, 0, 0, 1, 1)

        self.toolBox.addItem(self.page_2, u"\u5730\u8d28\u529b\u5b66\u9759\u6001\u8ba1\u7b97")

        self.verticalLayout.addWidget(self.toolBox)


        self.horizontalLayout.addWidget(self.groupBox)

        self.gb_main_left_area = QGroupBox(self.centralwidget)
        self.gb_main_left_area.setObjectName(u"gb_main_left_area")
        self.gridLayout_main_left = QGridLayout(self.gb_main_left_area)
        self.gridLayout_main_left.setSpacing(4)
        self.gridLayout_main_left.setObjectName(u"gridLayout_main_left")
        self.gridLayout_main_left.setContentsMargins(-1, -1, -1, 9)

        self.horizontalLayout.addWidget(self.gb_main_left_area)

        APPMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(APPMainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 731, 33))
        self.menu_file = QMenu(self.menubar)
        self.menu_file.setObjectName(u"menu_file")
        self.menu_subp = QMenu(self.menubar)
        self.menu_subp.setObjectName(u"menu_subp")
        self.menu_setting = QMenu(self.menubar)
        self.menu_setting.setObjectName(u"menu_setting")
        self.menu_help = QMenu(self.menubar)
        self.menu_help.setObjectName(u"menu_help")
        APPMainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(APPMainWindow)
        self.statusbar.setObjectName(u"statusbar")
        APPMainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_subp.menuAction())
        self.menubar.addAction(self.menu_setting.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())
        self.menu_file.addSeparator()
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_import_Ddata)
        self.menu_file.addAction(self.action_import_DTdata)
        self.menu_subp.addAction(self.action_subp1)
        self.menu_subp.addAction(self.action_subp2)
        self.menu_subp.addAction(self.action_subp3)
        self.menu_setting.addAction(self.action_conf)
        self.menu_help.addAction(self.action_helpDocs)

        self.retranslateUi(APPMainWindow)

        self.toolBox.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(APPMainWindow)
    # setupUi

    def retranslateUi(self, APPMainWindow):
        APPMainWindow.setWindowTitle(QCoreApplication.translate("APPMainWindow", u"\u4e95\u58c1\u7a33\u5b9a\u4f18\u5316\u7cfb\u7edf", None))
        self.actiondsad.setText(QCoreApplication.translate("APPMainWindow", u"dsad", None))
        self.actiondasd.setText(QCoreApplication.translate("APPMainWindow", u"dasd", None))
        self.actiondasd_2.setText(QCoreApplication.translate("APPMainWindow", u"dasd", None))
        self.action_import_Ddata.setText(QCoreApplication.translate("APPMainWindow", u"\u5bfc\u5165 \u5730\u8d28\u529b\u5b66\u6570\u636e", None))
        self.action_import_DTdata.setText(QCoreApplication.translate("APPMainWindow", u"\u5bfc\u5165 \u94bb\u4e95\u52a8\u6001\u6570\u636e", None))
        self.action_subp1.setText(QCoreApplication.translate("APPMainWindow", u"\u5730\u8d28\u529b\u5b66\u9759\u6001\u8ba1\u7b97", None))
        self.action_subp2.setText(QCoreApplication.translate("APPMainWindow", u"\u538b\u529b\u6ce2\u52a8\u52a8\u6001\u8ba1\u7b97", None))
        self.action_subp3.setText(QCoreApplication.translate("APPMainWindow", u"\u673a\u68b0\u6270\u52a8\u52a8\u6001\u8ba1\u7b97", None))
        self.action_conf.setText(QCoreApplication.translate("APPMainWindow", u"\u9879\u76ee\u8bbe\u7f6e", None))
        self.action_helpDocs.setText(QCoreApplication.translate("APPMainWindow", u"\u5e2e\u52a9\u6587\u6863", None))
        self.groupBox.setTitle("")
        self.groupBox_3.setTitle("")
        self.label.setText(QCoreApplication.translate("APPMainWindow", u"\u8d77\u59cb\u6df1\u5ea6 (m)", None))
        self.label_2.setText(QCoreApplication.translate("APPMainWindow", u"\u622a\u6b62\u6df1\u5ea6 (m)", None))
        self.pb_adjDepth.setText(QCoreApplication.translate("APPMainWindow", u"\u8c03\u6574", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), QCoreApplication.translate("APPMainWindow", u"\u6df1\u5ea6\u8c03\u6574", None))
        self.groupBox_4.setTitle("")
        self.pb_homorock.setText(QCoreApplication.translate("APPMainWindow", u"\u659c\u4e95\u574d\u584c\u538b\u529b\u5256\u9762\u8ba1\u7b97", None))
        self.pb_added_homorock.setText(QCoreApplication.translate("APPMainWindow", u"\u53e0\u52a0\u632f\u52a8\u574d\u584c\u538b\u529b\u8ba1\u7b97", None))
        self.pb_drilling_velocity.setText(QCoreApplication.translate("APPMainWindow", u"\u78b0\u649e\u901f\u5ea6\u4eff\u771f\u8ba1\u7b97", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), QCoreApplication.translate("APPMainWindow", u"\u5730\u8d28\u529b\u5b66\u9759\u6001\u8ba1\u7b97", None))
        self.gb_main_left_area.setTitle("")
        self.menu_file.setTitle(QCoreApplication.translate("APPMainWindow", u"\u6587\u4ef6", None))
        self.menu_subp.setTitle(QCoreApplication.translate("APPMainWindow", u"\u7a33\u5b9a\u6027\u5206\u6790", None))
        self.menu_setting.setTitle(QCoreApplication.translate("APPMainWindow", u"\u8bbe\u7f6e", None))
        self.menu_help.setTitle(QCoreApplication.translate("APPMainWindow", u"\u5e2e\u52a9", None))
    # retranslateUi

