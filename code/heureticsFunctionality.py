# -*- coding: utf-8 -*-
"""
@author: Emirhan Ak
"""

from clusteringFunctionality import ClusteringTab
import random 
import numpy as np
from numpy import exp
from itertools import combinations
import math as m

class HeureticsTab(ClusteringTab):
    
    def __init__(self, initialCanvas, finalCanvas, infoPanelTextBox, resultsTextBox):
        """
        Constructor of the HeureticsTab class. Initializes the ClusteringTab class.

        Parameters
        ----------
        initialCanvas : Initial solution canvas object.
        finalCanvas : Final solution canvas object.

        Returns
        -------
        None.

        """
        ClusteringTab.__init__(self, initialCanvas, finalCanvas, infoPanelTextBox)
        self.outputDataFrameData = None
        self.outputClosestPointToCenterDict = {}
        self.outputClusterCenterNode =None
        self.outputFurthestPointToHubDict = {}
        self.outputPairsOfObjectiveList = []
        self.finalSolutionData = None
        self.resultsTextBox = resultsTextBox
        self.flag = 0
        self.text = ""
        
    def setHeureticsData(self):
        """
        Function that sets the heuretics data that is calculated.

        Returns
        -------
        None.

        """
        self.outputDataFrameData = self.getClusteringData().copy(deep = True)
        self.outputClosestPointToCenterDict, self.outputClusterCenterNode = self.getClosestPoint()
        
    def redistributeHubs(self):
        """
        Function that reallocates the hub.

        Returns
        -------
        None.

        """
        randomHub  = random.choice(self.outputClusterCenterNode)
        cluster = self.outputDataFrameData.iloc[randomHub]["Labels"] 
        
        if self.outputDataFrameData[self.outputDataFrameData["Labels"] == cluster].index.size >= 2 :
            sampleNodeIndex = self.outputDataFrameData[self.outputDataFrameData["Labels"] == cluster].sample().index.to_list()
            
            while randomHub == sampleNodeIndex:
                randomHub  = random.choice(self.outputClusterCenterNode)
                cluster = self.outputDataFrameData.iloc[randomHub]["Labels"] 
                sampleNodeIndex = self.outputDataFrameData[self.outputDataFrameData["Labels"] == cluster].sample().index.to_list()
                
            temporary = self.outputDataFrameData.loc[randomHub,["X", "Y"]]
            self.outputDataFrameData.loc[randomHub,["X", "Y"]] = self.outputDataFrameData.loc[sampleNodeIndex[0], ["X", "Y"]]
            self.outputDataFrameData.loc[sampleNodeIndex[0],["X", "Y"]] = temporary
            
    def redistributeNodes(self):
        """
        Function that reallocates the node.

        Returns
        -------
        None.

        """
        randomNonHubList =  [i for i in self.outputDataFrameData.index.to_list() if i not in self.outputClusterCenterNode]
        randomNonHubNode = random.choice(randomNonHubList)
        labels = self.outputDataFrameData.loc[randomNonHubNode, "Labels"]
        self.outputDataFrameData.loc[randomNonHubNode, "Labels"] = np.random.choice(np.delete(self.outputDataFrameData["Labels"].values,labels))
            
    def changePlacesOfNodes(self):
        """
        Function that swaps the nodes.

        Returns
        -------
        None.

        """
        nonHubList1 =  [i for i in self.outputDataFrameData.index.to_list() if i not in self.outputClusterCenterNode]
        node1Index = random.choice(nonHubList1)
        nonHubList1.remove(node1Index)
        self.outputDataFrameData[self.outputDataFrameData["Labels"] == self.outputDataFrameData.loc[node1Index, "Labels"]]["Labels"].to_list()
        nonHubList2 = list(set(nonHubList1) - set(self.outputDataFrameData[self.outputDataFrameData["Labels"] == self.outputDataFrameData.loc[node1Index, "Labels"]].index.to_list()))
        node2Index = random.choice(nonHubList2)
        temporary = self.outputDataFrameData.loc[node1Index, ["X", "Y"]]
        self.outputDataFrameData.loc[node1Index,["X", "Y"]] = self.outputDataFrameData.loc[node2Index, ["X", "Y"]]
        self.outputDataFrameData.loc[node2Index,["X", "Y"]] = temporary
        
    def hillClimbingCalculation(self, numberOfIterations = 100):
        """
        Function that calculates the Hill Climbing of the opened data and displays it on final solution canvas.

        Parameters
        ----------
        numberOfIterations : The number of iterations that will be used by the Hill Climbing algorithm. The default is 100.

        Returns
        -------
        None.

        """
        self.resultsTextBox.setText("Hill Climbing Calculation Started!")
        self.setHeureticsData()
        solutionEvaluationData = self.getObjectiveFunctionResult()
        
        for i in range(numberOfIterations):
            k = np.random.randint(0, 3)
            if k == 0:
                self.redistributeHubs()
            elif k == 1:
                
                self.redistributeHubs()
            elif k == 2:
                self.changePlacesOfNodes()
                
            self.outputClosestPointCenter()
            self.outputFarthestHubDistance()
            self.outputObjectiveFunction()
            candidateEvaluationData = self.__finalObjectiveFunctionResult
            
            if candidateEvaluationData <= solutionEvaluationData:
                self.flag = 1
                solutionEvaluationData = candidateEvaluationData
                self.setFinalSolutionData(self.outputDataFrameData)
                self.resultsTextBox.append("Iteration:" + str(i))
                self.resultsTextBox.append("Result of Objective Function = " + str(solutionEvaluationData))
                
        if self.flag == 0:
            self.setFinalSolutionData(self.getClusteringData().copy(deep = True))
            self.resultsTextBox.append("Result of Objective Function = ")
            self.resultsTextBox.append(str(self.getObjectiveFunctionResult()))
            
        self.text = self.resultsTextBox.toPlainText()
        self.printFinalClusters()
        self.plotHub()
        self.flag = 0
        
    def simulatedAnnelingCalculation(self, numberOfIterations = 100, temporary = 10e-5):
        """
        Function that calculates the Simulated Anneling of the opened data and displays it on final solution canvas. 

        Parameters
        ----------
        numberOfIterations : The number of iterations that will be use by the Simulated Anneling algorithm. The default is 100.
        temporary : The temporary value that will be used by the Simulated Anneling algorithm. The default is 10e-5.

        Returns
        -------
        None.

        """
        self.resultsTextBox.setText("Simulated  Anneling Calculation Started!")
        self.setHeureticsData()
        bestEvaluation = self.getObjectiveFunctionResult()
        currentEvaluation = bestEvaluation
        
        for i in range(numberOfIterations):
            k = np.random.randint(0, 3)
            if k == 0:
                self.redistributeHubs()
            elif k == 1:
                self.redistributeHubs()
            else:
                self.changePlacesOfNodes()
                
            self.outputClosestPointCenter()
            self.outputFarthestHubDistance()
            self.outputObjectiveFunction()
            candidateEvaluation = self.__finalObjectiveFunctionResult
            
            if candidateEvaluation < bestEvaluation:
                self.flag = 1
                bestEvaluation = candidateEvaluation
                self.setFinalSolutionData(self.outputDataFrameData)
                self.resultsTextBox.append("Iteration:" + str(i))
                self.resultsTextBox.append("Result of Objective Function = " + str(bestEvaluation))

            differenceOfEvaluations = candidateEvaluation - currentEvaluation
            tVariable = temporary / float(i + 1)
            metropolisOfEvaluations = exp(-differenceOfEvaluations / tVariable)
            
            if differenceOfEvaluations < 0 or np.random.rand() < metropolisOfEvaluations:
                self.flag = 1
                currentEvaluation = candidateEvaluation
                bestEvaluation = candidateEvaluation
                self.setFinalSolutionData(self.outputDataFrameData)
                self.resultsTextBox.append("Iteration:" + str(i))
                self.resultsTextBox.append("Result of Objective Function = " + str(bestEvaluation))
        
        if self.flag == 0:
            self.setFinalSolutionData(self.getClusteringData().copy(deep = True))
            self.resultsTextBox.append("Result of Objective Function = ")
            self.resultsTextBox.append(str(self.getObjectiveFunctionResult()))
            
        self.printFinalClusters()
        self.plotHub()
        self.text = self.resultsTextBox.toPlainText()
        self.flag = 0
        
    def outputClosestPointCenter(self):
        """
        Function that determines the closest points to the center.

        Returns
        -------
        None.

        """
        self.outputClosestPointToCenterDict = {}
        self.outputClusterCenterNode =[]
        
        for i in range(0, self.outputDataFrameData["Labels"].nunique()):    
            point_id = self.outputDataFrameData[self.outputDataFrameData["Labels"] == i + 1].sub(self.outputDataFrameData[self.outputDataFrameData["Labels"] == i + 1].mean()).pow(2).sum(1).idxmin()
            self.outputClusterCenterNode.append(point_id)
            self.outputClosestPointToCenterDict[i + 1] = [self.outputDataFrameData["X"].loc[point_id], self.outputDataFrameData["Y"].loc[point_id]]

    def outputFarthestHubDistance(self):
        """
        Function that determines the furthest distances to the hub.

        Returns
        -------
        None.

        """
        self.outputFurthestPointToHubDict.clear()
        
        for i in range(0, self.outputDataFrameData["Labels"].nunique()):    
            distance = self.outputDataFrameData[self.outputDataFrameData["Labels"] == i + 1][["X", "Y"]].sub(self.outputClosestPointToCenterDict[i + 1]).pow(2).sum(1).pow(1 / 2)
            point_id = distance.idxmax()
            self.outputFurthestPointToHubDict[self.outputClusterCenterNode[i]] = distance.loc[point_id]
            
        self.outputOfAllPossiblePairs = [i for i in combinations(self.outputFurthestPointToHubDict.keys(), 2)]
    
    def outputObjectiveFunction(self):
        """
        Function that determines the output pairs of the objective function.

        Returns
        -------
        None.

        """
        self.outputPairsOfObjectiveList.clear()
        
        for i in self.outputOfAllPossiblePairs:
            temporaryObject = self.outputFurthestPointToHubDict[i[0]] + 0.75 * m.sqrt(self.outputDataFrameData.loc[i[0]][["X", "Y"]].sub(self.outputDataFrameData.loc[i[1]][["X", "Y"]]).pow(2).sum()) + self.outputFurthestPointToHubDict[i[1]]
            self.outputPairsOfObjectiveList.append(temporaryObject)
            
        self.outputPairsOfObjectiveList.append(2 * max(self.outputFurthestPointToHubDict.values()))
        self.__finalObjectiveFunctionResult = max(self.outputPairsOfObjectiveList)

    def printFinalClusters(self):
        """
        Function that displays the final clusters on the final solution canvas.

        Returns
        -------
        None.

        """
        self.finalCanvas.axes.cla()  
        self.finalCanvas.axes.scatter("X", "Y", c = "Labels", data = self.getFinalSolutionData())
        
        for i in self.getFinalSolutionData().index.to_list():
            self.finalCanvas.axes.text(self.getFinalSolutionData()["X"].iloc[i], self.getFinalSolutionData()["Y"].iloc[i], str(i))
            
        self.finalCanvas.draw()
    
    def plotHub(self):
        """
        Function that plots and displays the hubs on the final solution canvas.

        Returns
        -------
        None.

        """
        for i in range(0,self.getFinalSolutionData()["Labels"].nunique()):    
            point_id = self.getFinalSolutionData()[self.getFinalSolutionData()["Labels"] == i + 1].sub(self.getFinalSolutionData()[self.getFinalSolutionData()["Labels"] == i + 1].mean()).pow(2).sum(1).idxmin()
            self.finalCanvas.axes.scatter("X", "Y", c = '#fc00fc', data = self.getFinalSolutionData().loc[point_id])
            self.finalCanvas.draw()
            
    def setFinalSolutionData(self, finalSolutionData):
        """
        Function that sets the final solution data.

        Parameters
        ----------
        finalSolutionData : Final solution data after the heuretics operations.

        Returns
        -------
        None.

        """
        self.finalSolutionData = finalSolutionData
    
    def getFinalSolutionData(self):
        """
        Function that returns the final solution data.

        Returns
        -------
        Final solution data.

        """
        return self.finalSolutionData
        
    def setResultTextBox(self):
        """
        Function that sets the text on the TextEdit object associated with the results.

        Returns
        -------
        None.

        """
        self.resultsTextBox.setText(self.text)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        