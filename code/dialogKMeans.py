# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogKMeans.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialogKMeans(object):
    def setupUi(self, dialogKMeans):
        dialogKMeans.setObjectName("dialogKMeans")
        dialogKMeans.resize(268, 247)
        self.gridLayout = QtWidgets.QGridLayout(dialogKMeans)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(dialogKMeans)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.dialogKMeans_nClusterTextBox = QtWidgets.QTextEdit(dialogKMeans)
        self.dialogKMeans_nClusterTextBox.setObjectName("dialogKMeans_nClusterTextBox")
        self.gridLayout.addWidget(self.dialogKMeans_nClusterTextBox, 0, 1, 1, 2)
        self.label_2 = QtWidgets.QLabel(dialogKMeans)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.dialogKMeans_initComboBox = QtWidgets.QComboBox(dialogKMeans)
        self.dialogKMeans_initComboBox.setObjectName("dialogKMeans_initComboBox")
        self.dialogKMeans_initComboBox.addItem("")
        self.dialogKMeans_initComboBox.addItem("")
        self.gridLayout.addWidget(self.dialogKMeans_initComboBox, 1, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(dialogKMeans)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 2)
        self.dialogKMeans_max_iterTextBox = QtWidgets.QTextEdit(dialogKMeans)
        self.dialogKMeans_max_iterTextBox.setObjectName("dialogKMeans_max_iterTextBox")
        self.gridLayout.addWidget(self.dialogKMeans_max_iterTextBox, 2, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(dialogKMeans)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.dialogKMeans_algorithmComboBox = QtWidgets.QComboBox(dialogKMeans)
        self.dialogKMeans_algorithmComboBox.setObjectName("dialogKMeans_algorithmComboBox")
        self.dialogKMeans_algorithmComboBox.addItem("")
        self.dialogKMeans_algorithmComboBox.addItem("")
        self.dialogKMeans_algorithmComboBox.addItem("")
        self.gridLayout.addWidget(self.dialogKMeans_algorithmComboBox, 3, 1, 1, 2)
        self.buttonBox = QtWidgets.QDialogButtonBox(dialogKMeans)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 0, 1, 3)

        self.retranslateUi(dialogKMeans)
        self.buttonBox.accepted.connect(dialogKMeans.accept) # type: ignore
        self.buttonBox.rejected.connect(dialogKMeans.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(dialogKMeans)

    def retranslateUi(self, dialogKMeans):
        _translate = QtCore.QCoreApplication.translate
        dialogKMeans.setWindowTitle(_translate("dialogKMeans", "K-Means"))
        self.label.setText(_translate("dialogKMeans", "Number of clusters, int, Default value is 8"))
        self.dialogKMeans_nClusterTextBox.setHtml(_translate("dialogKMeans", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">8</p></body></html>"))
        self.label_2.setText(_translate("dialogKMeans", "Initialization Method, Default value is k-means++"))
        self.dialogKMeans_initComboBox.setItemText(0, _translate("dialogKMeans", "k-means++"))
        self.dialogKMeans_initComboBox.setItemText(1, _translate("dialogKMeans", "random"))
        self.label_3.setText(_translate("dialogKMeans", "Maximum Number of Iterations, int, Default value is 300"))
        self.dialogKMeans_max_iterTextBox.setHtml(_translate("dialogKMeans", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">300</p></body></html>"))
        self.label_4.setText(_translate("dialogKMeans", "Algorithm Type, Default value is auto"))
        self.dialogKMeans_algorithmComboBox.setItemText(0, _translate("dialogKMeans", "auto"))
        self.dialogKMeans_algorithmComboBox.setItemText(1, _translate("dialogKMeans", "elkan"))
        self.dialogKMeans_algorithmComboBox.setItemText(2, _translate("dialogKMeans", "full"))
