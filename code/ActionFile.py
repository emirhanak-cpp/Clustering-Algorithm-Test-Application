# -*- coding: utf-8 -*-
"""
@author: Emirhan Ak
"""

from PyQt5.QtWidgets import QFileDialog
import numpy as np

class FileTab:
    
    def __init__(self, initialCanvas, finalCanvas):
        """
        Constructor of the FileTab Class. Initalizes canvas variables.

        Parameters
        ----------
        initialCanvas : Initial solution canvas object
        finalCanvas : Final solution canvas object

        Returns
        -------
        None.

        """
        self.__randomData = None
        self.__initialData = None
        self.__finalData = None
        self.initialCanvasObject = initialCanvas
        self.finalCanvas = finalCanvas
        
    def openData(self):
        """
        Function that opens data from (*.txt) file and displays it on the solution canvas.

        Returns
        -------
        None.

        """
        openDataPath,_ = QFileDialog.getOpenFileName(caption = "Open Data", filter = "Text Files (*.txt)")
        self.__randomData = np.loadtxt(openDataPath)
        self.initialCanvasObject.axes.cla()  
        self.initialCanvasObject.axes.scatter(self.__randomData[:,0] , self.__randomData[:,1],c = "#000000")
        for i in range(self.__randomData.shape[0]):
            self.initialCanvasObject.axes.text(self.__randomData[i,0], self.__randomData[i,1], str(i))
        self.initialCanvasObject.draw()
        
    def saveInitialSolutionAsTxt(self):
        """
        Function that saves the inital solution as (*.txt) file.

        Returns
        -------
        None.

        """
        self.setInitialData(self.__randomData)
        openDataPath, openFileName = QFileDialog.getSaveFileName(caption = "Save Initial Data", filter = "Text Files (*.txt)")
        np.savetxt(openDataPath, self.__randomData)
        
    def saveFinalSolutionAsTxt(self, finalSolutionData):
        """
        Function that saves the final solution as (*.txt) file.

        Parameters
        ----------
        finalSolutionData : Data that is displayed on the final solution canvas.
        
        Returns
        -------
        None.

        """
        saveDataPath, saveFileName = QFileDialog.getSaveFileName(caption = "Save Final Data", filter = "Text Files (*.txt)")
        self.__finalData = finalSolutionData[["X", "Y", "Labels"]].values
        np.savetxt(saveDataPath, self.__finalData)
        
    def exportInitialSolutionAsTxtOrJpg(self):
        """
        Function that exports initial solution as (*.txt) or (*.jpg) file.

        Returns
        -------
        None.

        """
        exportDataPath,_ = QFileDialog.getSaveFileName(caption = "Export Initial Data", filter= "Images Files (*.jpg);; Text Files (*.txt)")
        self.initialCanvasObject.print_figure(exportDataPath)
        
    def exportFinalSolutionAsTxtOrJpg(self):
        """
        Function that exports final solution as (*.txt) or (*.jpg) file.

        Returns
        -------
        None.

        """
        exportDataPath,_ = QFileDialog.getSaveFileName(caption = "Export Final Data", filter= "Images Files (*.jpg);; Text Files (*.txt)")
        self.finalCanvas.print_figure(exportDataPath)
        
    def plotRandomData(self, randomData):
        """
        Function that plots either the inital or the final data that it receives.

        Parameters
        ----------
        randomData : Initial or final data is desired to be displayed.

        Returns
        -------
        None.

        """
        self.initalCanvas.axes.cla()  
        self.initalCanvas.axes.scatter(randomData[:,0] , randomData[:,1])
        self.initalCanvas.draw()
        
    def getInitialCanvasObject(self):
        """
        Function that returns the initial solution canvas object.

        Returns
        -------
        Initial solution canvas.

        """
        return self.initialCanvasObject
    
    def setRandomData(self, randomData):
        """
        Function thats sets the temporary inital or final data.

        Parameters
        ----------
        randomData : Data that is stored temporarily. Can either be initial or final data.

        Returns
        -------
        None.

        """
        self.__randomData = randomData
        
    def getRandomData(self):
        """
        Function that returns the temporary data that is eiher inital or final.

        Returns
        -------
        Temporary data that is either inital or final.

        """
        return self.__randomData
    
    def setInitialData(self, initialData):
        """
        Function that sets the inital solution data.

        Parameters
        ----------
        initialData : Initial data that is desired to be passed into the object.

        Returns
        -------
        None.

        """
        self.__initialData = initialData
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        