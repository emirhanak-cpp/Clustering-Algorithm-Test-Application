# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogMeanShift.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialogMeanShift(object):
    def setupUi(self, dialogMeanShift):
        dialogMeanShift.setObjectName("dialogMeanShift")
        dialogMeanShift.resize(259, 221)
        self.gridLayout = QtWidgets.QGridLayout(dialogMeanShift)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(dialogMeanShift)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.dialogMeanShift_bandwidthTextBox = QtWidgets.QTextEdit(dialogMeanShift)
        self.dialogMeanShift_bandwidthTextBox.setObjectName("dialogMeanShift_bandwidthTextBox")
        self.gridLayout.addWidget(self.dialogMeanShift_bandwidthTextBox, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(dialogMeanShift)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.dialogMeanShift_bin_seedingComboBox = QtWidgets.QComboBox(dialogMeanShift)
        self.dialogMeanShift_bin_seedingComboBox.setObjectName("dialogMeanShift_bin_seedingComboBox")
        self.dialogMeanShift_bin_seedingComboBox.addItem("")
        self.dialogMeanShift_bin_seedingComboBox.addItem("")
        self.gridLayout.addWidget(self.dialogMeanShift_bin_seedingComboBox, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(dialogMeanShift)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.dialogMeanShift_max_iterTextBox = QtWidgets.QTextEdit(dialogMeanShift)
        self.dialogMeanShift_max_iterTextBox.setObjectName("dialogMeanShift_max_iterTextBox")
        self.gridLayout.addWidget(self.dialogMeanShift_max_iterTextBox, 2, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(dialogMeanShift)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 3, 0, 1, 2)

        self.retranslateUi(dialogMeanShift)
        self.buttonBox.accepted.connect(dialogMeanShift.accept) # type: ignore
        self.buttonBox.rejected.connect(dialogMeanShift.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(dialogMeanShift)

    def retranslateUi(self, dialogMeanShift):
        _translate = QtCore.QCoreApplication.translate
        dialogMeanShift.setWindowTitle(_translate("dialogMeanShift", "Mean-shift"))
        self.label.setText(_translate("dialogMeanShift", "Bandwidth, float, Default value is None"))
        self.dialogMeanShift_bandwidthTextBox.setHtml(_translate("dialogMeanShift", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">None</p></body></html>"))
        self.label_3.setText(_translate("dialogMeanShift", "Enable Seeding, bool, Default value is False"))
        self.dialogMeanShift_bin_seedingComboBox.setItemText(0, _translate("dialogMeanShift", "False"))
        self.dialogMeanShift_bin_seedingComboBox.setItemText(1, _translate("dialogMeanShift", "True"))
        self.label_2.setText(_translate("dialogMeanShift", "Maximum Number of Iterations, int, Default value is 300"))
        self.dialogMeanShift_max_iterTextBox.setHtml(_translate("dialogMeanShift", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">300</p></body></html>"))
