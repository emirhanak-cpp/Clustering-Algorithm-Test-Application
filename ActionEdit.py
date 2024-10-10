# -*- coding: utf-8 -*-
"""
@author: Emirhan Ak
"""

from ActionFile import FileTab

class EditTab(FileTab):
    
    def __init__(self, initialCanvas, finalCanvas, infoPanelTextBox):
        """
        Constructor of the EditTab class. Initializes the FileTab class that it inherits.

        Parameters
        ----------
        initialCanvas : Initial solution canvas object.
        finalCanvas : Final solution canvas object.

        Returns
        -------
        None.

        """
        FileTab.__init__(self, initialCanvas, finalCanvas)
        self.setInformationPanelObject(infoPanelTextBox)
        
    def clearInitialData(self):
        """
        Function that clears the inital solution canvas and initial solution data.

        Returns
        -------
        None.

        """
        self.initialCanvasObject = self.getInitialCanvasObject()
        self.initialCanvasObject.axes.cla()
        self.initialCanvasObject.draw()
        self.setRandomData(None)
        self.infoPanelTextBoxObject.setText("Initial Data Cleared.")
        
    def clearFinalData(self):
        """
        Function that clears the final solution canvas and final solution data.

        Returns
        -------
        None.

        """
        self.finalCanvas.axes.cla()
        self.finalCanvas.draw()
    
    def setInformationPanelObject(self, infoPanelTextBox):
        """
        Function that sets the TextEdit object for the information panel

        Parameters
        ----------
        infoPanelTextBox : TextEdit object for the information panel

        Returns
        -------
        None.

        """
        self.infoPanelTextBoxObject = infoPanelTextBox