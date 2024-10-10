# -*- coding: utf-8 -*-
"""
@author: Emirhan Ak
"""

from ActionEdit import EditTab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, DBSCAN, AgglomerativeClustering
from itertools import combinations
import math as m
import warnings
warnings.filterwarnings("ignore")

class ClusteringTab(EditTab):
    
    def __init__(self, initialCanvas, finalCanvas, infoPanelTextBox):
        """
        Constructor of the ClusteringTab class. Initializes the EditTab class that it inherits.

        Parameters
        ----------
        initialCanvas : Initial solution canvas object.
        finalCanvas : Final solution canvas object.

        Returns
        -------
        None.

        """
        EditTab.__init__(self, initialCanvas, finalCanvas, infoPanelTextBox)
        self.__label = None
        self.__randomData = None
        self.__closestPointToCenterDict = {}
        self.__furthestPointToHubDict = {}
        self.__pairsOfObjectiveList = []
        
    def kMeansCalculation(self, numberOfClusters = 8, initializerMethod = "k-means++", iterationNumber = 300, algorithmType = "auto"):
        """
        Function that calculates the KMeans of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        numberOfClusters : The number of clusters that will be used by the KMeans algorithm. The default is 8.
        initializerMethod : The initialization method that will be used by the KMeans algorithm. The default is "k-means++".
        iterationNumber : The iteration number that will be used by the KMeans algorithm. The default is 300.
        algorithmType : The algortihm type that will be used by the KMeans algorithm. The default is "auto".

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData = KMeans(n_clusters = numberOfClusters, init = initializerMethod, max_iter = iterationNumber, algorithm = algorithmType).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__centerPoint = pd.DataFrame(self.__clusteringData.cluster_centers_, columns = ["X", "Y"])
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData["Labels"] = self.__label + 1
        self.doGeneralWork()
        
    def affinityProagationCalculation(self, dampingAmount = 0.5, iterationNumber = 200, convergenceIteration = 15, affinityType = "euclidean" ):
        """
        Function that calculates the Affinity Propagation of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        dampingAmount : The damping amount that will be used by the Affinity Propagation algorithm. The default is 0.5.
        iterationNumber : The iteration number that will be used by the Affinity Propagation algorithm. The default is 200.
        convergenceIteration : The convergence iteration number that will be used by the Affinity Propagation algorithm. The default is 15.
        affinityType : The type of affinity that will be used by the Affinity Propagation algorithm. The default is "euclidean".

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData  = AffinityPropagation(damping = dampingAmount, max_iter = iterationNumber, convergence_iter = convergenceIteration).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__centerPoint = pd.DataFrame(self.__clusteringData.cluster_centers_, columns = ["X", "Y"])
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData["Labels"] = self.__label + 1
        self.doGeneralWork()
        
    def meanShiftCalculation(self, bandwidthValue = None, seedingsFlag = False, iterationNumber = 300):
        """
        Function that calculates the Mean-Shift of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        bandwidthValue : The bandwidth that will be used by the Mean-Shift algorithm. The default is None.
        seedingsFlag : The seedings flag that will be used by the Mean-Shift algorithm. The default is False.
        iterationNumber : The iteration number that will be used by the Mean-Shift algorithm. The default is 300.

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData = MeanShift(bandwidth = bandwidthValue, bin_seeding = seedingsFlag, max_iter = iterationNumber).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__centerPoint = pd.DataFrame(self.__clusteringData.cluster_centers_,columns = ["X", "Y"])
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData["Labels"] = self.__label + 1
        self.doGeneralWork()
        
    def spectralClusteringCalculation(self, numberOfClusters = 8, numberOfNeighbors = 10, affinityType = "rbf", initilazierNumber = 10, labelToAssign = "kmeans"):
        """
        Function that calculates the Spectral Clustering of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        numberOfClusters : The number of clusters that will be used by the Spectral Clustering algorithm. The default is 8.
        numberOfNeighbors : The number of neighbors that will be used by the Spectral Clustering algorithm. The default is 10.
        affinityType : The affinity type that will be used by the Spectral Clustering algorithm. The default is "rbf".
        initilazierNumber : The initialization number that will be used by the Spectral Clustering algorithm. The default is 10.
        labelToAssign : The label that will be assigned by the Spectral Clustering algorithm The default is "kmeans".

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData = SpectralClustering(n_clusters = numberOfClusters, n_neighbors = numberOfNeighbors, n_init = initilazierNumber, assign_labels = labelToAssign, affinity = affinityType).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData ["Labels"] = self.__label + 1
        self.__centerPoint = pd.DataFrame(self.__dataFrameData.groupby("Labels")[["X", "Y"]].mean())
        self.doGeneralWork()
        
    def hierarchialClusteringCalculation(self, numberOfClusters = 2, affinityType = "euclidean", linkageType = "ward"):     
        """
        Function that calculates the Hierarchial Clustering of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        numberOfClusters : The number of clusters that will be used by the Hierarchial Clustering. The default is 2.
        affinityType : The type of affinity that will be used by the Hierarchial Clustering. The default is "euclidean".
        linkageType : The linkage type that will be used by the Hierarchial Clustering. The default is "ward".

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData = AgglomerativeClustering(n_clusters = numberOfClusters, affinity = affinityType, linkage = linkageType).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__label = self.__clusteringData.labels_
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData["Labels"] = self.__label + 1
        self.__centerPoint = pd.DataFrame(self.__dataFrameData.groupby("Labels")[["X", "Y"]].mean())
        self.doGeneralWork()
        
    def DBSCANCalculation(self, epsValue = 0.5, minimumNumberOfSamples = 5, algorithmType = "auto"):     
        """
        Function that calculates the DBSCAN of the opened data and displays it on the inital solution canvas.

        Parameters
        ----------
        epsValue : The eps value that will be used by the DBSCAN. The default is 0.5.
        minimumNumberOfSamples : The minimum number of samples that will be used by the DBSCAN. The default is 5.
        algorithmType : The algorithm type that will be applied by the DBSCAN. The default is "auto".

        Returns
        -------
        None.

        """
        self.__randomData = self.getRandomData()
        self.__clusteringData = DBSCAN(eps = epsValue, min_samples = minimumNumberOfSamples, algorithm = algorithmType).fit(self.__randomData)
        self.__label = self.__clusteringData.labels_
        self.__label = self.__clusteringData.labels_
        self.__dataFrameData = pd.DataFrame(data = self.__randomData, columns = ["X", "Y"])
        self.__dataFrameData ["Labels"] = self.__label + 1  
        self.__centerPoint = pd.DataFrame(self.__dataFrameData.groupby("Labels")[["X","Y"]].mean())
        self.doGeneralWork()
        
    def printClusterOnInitialCanvas(self):
        """
        Function that displays the cluster on inital solution canvas.

        Returns
        -------
        None.

        """
        self.infoPanelTextBoxObject.setPlainText("Labels of Points\n" + str(self.__label + 1))
        self.infoPanelTextBoxObject.append("\nThe number of clusters are " + str(self.__dataFrameData["Labels"].nunique())) 
        
        for i in range(0, self.__dataFrameData["Labels"].nunique()):
            self.infoPanelTextBoxObject.append("Cluster Label " + str(i + 1) + " >>>>>" + str(self.__dataFrameData[self.__dataFrameData["Labels"] == i + 1].index.to_list()))
            
        self.initialCanvasObject.axes.cla()  
        self.initialCanvasObject.axes.scatter("X", "Y", c = "Labels", data = self.__dataFrameData)
        self.initialCanvasObject.axes.scatter("X", "Y", c = '#fc00fc', label = "Center", data = self.__centerPoint, marker = "D")
        
        for i in self.__dataFrameData.index.to_list():
            self.initialCanvasObject.axes.text(self.__dataFrameData["X"].iloc[i], self.__dataFrameData["Y"].iloc[i], str(i))
        
        self.initialCanvasObject.draw()
        
        plt.scatter("X", "Y", c = "Labels", data = self.__dataFrameData)
        plt.scatter("X", "Y", data = self.__centerPoint, marker = "D")
        
    def closestPointToCenterCalculation(self):
        """
        Function that determines the closest point to the center of each cluster.

        Returns
        -------
        None.

        """
        self.clusterCenterNodeList = []
        
        for i in range(0, self.__dataFrameData["Labels"].nunique()):    
            pointID = self.__dataFrameData[self.__dataFrameData["Labels"] == i + 1].sub(self.__dataFrameData[self.__dataFrameData["Labels"] == i + 1].mean()).pow(2).sum(1).idxmin()
            self.initialCanvasObject.axes.scatter("X","Y", c = '#fc00fc', data = self.__dataFrameData.loc[pointID])
            self.initialCanvasObject.draw()
            plt.scatter("X", "Y", c = '#fc00fc', data = self.__dataFrameData.loc[pointID])
            self.clusterCenterNodeList.append(pointID)
            self.__closestPointToCenterDict[i + 1] = [self.__dataFrameData["X"].loc[pointID], self.__dataFrameData["Y"].loc[pointID]]
            
        self.infoPanelTextBoxObject.append("Center Node of Each Cluster >>>>>" + str(self.clusterCenterNodeList))
        self.infoPanelTextBoxObject.append("Closest Point to Center Point of Each Cluster:\n" + str(self.__closestPointToCenterDict))
        
    def furthestDistanceToHubCalculation(self):
        """
        Function that determines the farthest distance to each cluster.

        Returns
        -------
        None.

        """
        self.__furthestPointToHubDict.clear()
        #self.infoPanelTextBoxObject.append("\n\nFurthest Hub Distances:")
        
        for i in range(0, self.__dataFrameData["Labels"].nunique()):    
            distance = self.__dataFrameData[self.__dataFrameData["Labels"] == i + 1][["X", "Y"]].sub(self.__closestPointToCenterDict[i + 1]).pow(2).sum(1).pow(1/2)
            pointID = distance.idxmax()
            self.__furthestPointToHubDict[self.clusterCenterNodeList[i]] = distance.loc[pointID]
            
        #self.infoPanelTextBoxObject.append(str(self.__furthestPointToHubDict))
        self.allPossibleObjectivePairs = [i for i in combinations(self.__furthestPointToHubDict.keys(), 2)]
        #self.infoPanelTextBoxObject.append("All Pairs: " + str(self.allPossibleObjectivePairs))
        
    def objectiveFunctionResultCalculation(self):
        """
        Function that determines the pair objective list of each cluster.

        Returns
        -------
        None.

        """
        #self.infoPanelTextBoxObject.append("\n\nPair Objectives:")
        self.__pairsOfObjectiveList.clear()
        
        for i in self.allPossibleObjectivePairs:
            tempObject = self.__furthestPointToHubDict[i[0]] + 0.75 * m.sqrt(self.__dataFrameData.loc[i[0]][["X", "Y"]].sub(self.__dataFrameData.loc[i[1]][["X","Y"]]).pow(2).sum()) + self.__furthestPointToHubDict[i[1]]
            self.__pairsOfObjectiveList.append(tempObject)
            
        self.__pairsOfObjectiveList.append(2*max(self.__furthestPointToHubDict.values()))
        #self.infoPanelTextBoxObject.append(str(self.__pairsOfObjectiveList))
        self.__initalObjectiveFunctionResult = max(self.__pairsOfObjectiveList)
        self.infoPanelTextBoxObject.append("Objective Function Result -----> {}".format(max(self.__pairsOfObjectiveList)))
        
    def doGeneralWork(self):
        """
        Function that handles general work that is included in all clustering calculations.

        Returns
        -------
        None.

        """
        self.printClusterOnInitialCanvas()
        self.closestPointToCenterCalculation()
        self.furthestDistanceToHubCalculation()
        self.objectiveFunctionResultCalculation()
        
    def getCluster(self):
        """
        Function that returns a cluster.

        Returns
        -------
        Cluster.

        """
        return self.__clustering
    
    def getLabel(self):
        """
        Function that return the label for a cluster.

        Returns
        -------
        Label for a cluster.

        """
        return self.__label
    
    def getCenter(self):
        """
        Function that returns the center point of a cluster.

        Returns
        -------
        Center point of a cluster.

        """
        return self.__centerPoint
    
    def getClusteringData(self):
        """
        Function that returns the clustering data of a cluster.

        Returns
        -------
        Clustering data of a cluster.

        """
        return self.__dataFrameData
    
    def getClosestPoint(self):
        """
        Function that returns the all of the closest points to the center of each cluster.

        Returns
        -------
        All of the closest points to the center of each cluster.

        """
        return self.__closestPointToCenterDict, self.clusterCenterNodeList
    
    def getFarthestPoint(self):
        """
        Function that returns all of the furthest points for each cluster.

        Returns
        -------
        All of the furthest points for each cluster.

        """
        return self.__furthestPointToHubDict
    
    def gePairObjective(self):
        """
        Function that returns all of the pair objectives for each cluster.

        Returns
        -------
        All of the pair objectives for each cluster.

        """
        return self.__pairsOfObjectiveList
    
    def getObjectiveFunctionResult(self):
        """
        Function that return the inital solution objective result.

        Returns
        -------
        Inital solution objective result.

        """
        return self.__initalObjectiveFunctionResult
    
    def getInformationPanel(self):
        """
        Function that return the information panel TextEdit object.

        Returns
        -------
        TextEdit object for the information panel.

        """
        return self.infoPanelTextBoxObject
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        