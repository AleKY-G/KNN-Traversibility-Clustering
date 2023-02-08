'''
K-Nearest Neighbors Traversal Clustering (KNNTC)

Test Datasets:
    - make_blobs (differing densities)
    - make_circles
    - make_moons

Algorithm Pseudocode:
1. For each point in P, which is the supplied dataset, assign pointsDict a new key.
    a. The value of the key is a list containing the cluster number of the point (which is currently null) and a list of all points P ordered by distance to point.
        - If multiple points in P share the same distance to point, then they are ordered by lowest X-coordinate value, then lowest Y-coordinate value.
2. Set currentPoint to the leftmost and lowest point, currentCluster to 0, and maximumClusterNumber to 0.
3. Initialize clusters[clusterNumber] to an empty list
4. While P is not empty:
    a. Remove currentPoint from P and append currentPoint to clusters[clusterNumber].
    b. Re-assign null cluster number of currentPoint to clusterNumber.
    c. Check if P is empty; if so, break out from loop.
    d. Set previousPoint to currentPoint.
    e. For neighbor in currentPoint's k-nearest neighbors.
        - Check if the neighbor is already clustered. If it is not, set currentPoint to neighbor and break out from the nested loop.
    f. If currentPoint is equal to previousPoint
        - Iterate through all of currentPoint’' neighbors. 
            -If currentPoint's neighbor is unclustered and has a k-nearest neighbor that is already assigned to ANY cluster, then set currentPoint to that neighbor and currentCluster to that neighbor’s cluster. Then, break out of the nested loop.
    g. If currentPoint still is equal to previousPoint (that means all the remaining unclustered points did not have a k-nearest neighbor that was already clustered).
        -Pick the closest unclustered neighbor to currentPoint, and set currentPoint to that neighbor. 
        -A new cluster needs to be created, so increment maximumClusterNumber by 1 and set clusterNumber to maximumClusterNumber
'''

import numpy as np
import math
import sys
import copy
from sklearn import datasets  
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

#Class used to plot the cluster
class Graph:
    def __init__(self):
        self.points = []
        self.colors = []
        self.outlierColors = []
        self.outliers = []

    #graphing regular points that were added to a cluster
    def addPoint(self, point, color):
        self.points.append(point)
        self.colors.append(color)

    #graphing outliers if they were removed
    def addOutlier(self, point, color):
        self.outliers.append(point)
        self.outlierColors.append(color)

    #resetting graph
    def clearGraph(self):
        self.points = []
        self.colors = []
        self.outlierColors = []
        self.outliers = []
    
    #saving to drive instead of displyaing
    def save(self, name):
        plt.clf()
        plt.scatter(*zip(*self.points), c=self.colors)
        #if there are any outliers, graph them
        if len(self.outliers) > 0:
            plt.scatter(*zip(*self.outliers), c=self.outlierColors)
        plt.savefig(name)

    #showing graph through matplotlib
    def show(self):
        plt.clf()
        plt.scatter(*zip(*self.points), c=self.colors)
        if len(self.outliers) > 0:
            plt.scatter(*zip(*self.outliers), c=self.outlierColors)
        plt.show()

#Class used to run the KNNTC
class KNNTC:
    def __init__(self, points,labels=None):
        #labels are the ground truths and are only used for testing purposes
        self.points = points
        self.originalPoints = copy.deepcopy(points)
        self.distancesDict = {}
        self.distances = []
        self.outliers = []
        self.clusters = {}
        self.pointsDict = {}
        self.graph = Graph()
        self.originalLabels = labels
        self.clusteredLabels = []
        self.maxClusterNumber = 0


    #finding the closest point to each point in the dataset
    def findDistances(self):
        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        for p in self.points:
            d = sys.maxsize
            for p2 in self.points:
                if distance(p, p2) < d and distance(p, p2) != 0:
                    d = distance(p, p2)
            self.distancesDict[p] = d
            self.distances.append(d)
        self.distancesDict = {k: v for k, v in sorted(self.distancesDict.items(), key=lambda item: item[1], reverse=True)}
        print("Distances Dict", self.distancesDict)
        print()


    #removing points whose nearest neighbor is too far away
    def removeOutliers(self, cutoff=1.5):
        #find outliers in distances
        outliers = []
        self.distances.sort()
        q1 = math.ceil(0.25 * len(self.distances))
        q3 = math.ceil(0.75 * len(self.distances))
        iqr = self.distances[q3] - self.distances[q1]

        #no upper bound because we only care about points that are too far away
        upperBound = self.distances[q3] + (cutoff * iqr)

        for i in range(len(self.distances)):
            if self.distances[i] > upperBound:
                outliers.append(self.distances[i])
        
        key_list= [k for k in self.distancesDict.keys()]
        val_list = [v for v in self.distancesDict.values()]

        #remove outliers from points
        for o in outliers:
            if key_list[val_list.index(o)] in self.points:
                self.points.remove(key_list[val_list.index(o)])
                self.outliers.append(key_list[val_list.index(o)])
                self.distances.remove(o)
                del self.distancesDict[key_list[val_list.index(o)]]

    #euclidean distance
    def distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def neighborsOrdered(self, p, points):
        #order neighbors by distance from p. If there are multiple points with the same distance, order them by leftmost point, and if they share the same x, then order them by lowest point
        neighbors = []
        for p2 in points:
            if p2 != p:
                neighbors.append(p2)
        neighbors.sort(key=lambda point: (self.distance(p, point), point[0], point[1]))
        return neighbors

    #making clusters from pre-processed point list
    def cluster(self, k, startingPoint = None):
        if startingPoint == None:
            #find the leftmost point; if there are multiple, find the lowest one
            startingPoint = self.points[0]
            for p in self.points:
                if p[0] < startingPoint[0]:
                    startingPoint = p
                elif p[0] == startingPoint[0]:
                    if p[1] < startingPoint[1]:
                        startingPoint = p
            print("Starting Point:", startingPoint)
        
        #populating pointsDict, which will be used to order the points in accordance to distance from the neighbors
        self.pointsCopy = copy.deepcopy(self.points)
        for point in self.pointsCopy:
            neighbors = self.neighborsOrdered(point, self.pointsCopy)
            self.pointsDict[point] = [neighbors, None]
        
        #setting currentPoint to startingPoint
        currentPoint = startingPoint
        clusterNumber = 0
        self.maxClusterNumber = 0

        #initializing first cluster as empty
        self.clusters[clusterNumber] = []

        #while there are still points to be added to clusters
        totalLen = len(self.pointsCopy)
        clustered = 0
        while clustered < totalLen: 
            #remove the current point from the available points and add it to the current cluster
            #self.pointsCopy.remove(currentPoint)
            self.clusters[clusterNumber].append(currentPoint)
            self.pointsDict[currentPoint][1] = clusterNumber
            clustered += 1
            print("Added " + str(currentPoint) + " to cluster " + str(clusterNumber))

            #used for checking if there are available neighbors to be added to the cluster
            prevPoint = currentPoint

            #in the event that there was 1 point left
            if totalLen == clustered:#len(self.pointsCopy) == 0:
                break
            
            #iterate through the currentPoint's k nearest neighbors
            for i in range (k):
                neighbor = self.pointsDict[currentPoint][0][i]
                if self.pointsDict[neighbor][1] is None:#and neighbor in self.pointsCopy:
                    #if it finds a k nearest neighbor that is unclustered, then set the current point to that neighbor and break out of the loop
                    currentPoint = neighbor
                    break

            flag1 = True
            
            #if currentPoint remains unchanged, meaning that all of the k nearest neighbors were already in a cluster
            if currentPoint == prevPoint:
                print("All of the neighbors of ", currentPoint, " are already in a cluster. Searching for a new current point...")
                #iterate through all of the currentPoint's neighbors
                for currPointNeighbor in self.pointsDict[currentPoint][0]:
                    #if currPointNeighbor has a nearest neighbor that is assigned to a cluster and currPointNeighbor is not assigned to a cluster:
                    if self.pointsDict[currPointNeighbor][1] is None:
                        for j in range(k):
                            if self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1] is not None and flag1 == True:
                                currentPoint = currPointNeighbor
                                print("New current point is ", currentPoint, " and cluster number is ", self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1], " because of ",self.pointsDict[currPointNeighbor][0][j], "")
                                clusterNumber = self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1]
                                flag1 = False

            #if currentPoint STILL remains unchanged, that means that there were no unclustered points that had a k nearest neighbor that was already in a cluster
            #so we just pick the first unclustered neighbor, and start a new cluster
            if currentPoint == prevPoint or clusterNumber is None:
                print("All neighbors (that are not the k nearest neighbors of) " + str(currentPoint) + "are unclustered. Choosing a new current point...")
                newPoint = None
                j = k
                while newPoint is None:
                    newPoint = self.pointsDict[currentPoint][0][j] if self.pointsDict[self.pointsDict[currentPoint][0][j]][1] is None else None
                    j+=1 
                currentPoint = newPoint
                self.maxClusterNumber += 1
                clusterNumber = self.maxClusterNumber
                print("New cluster point is ", currentPoint, " and cluster number is ", clusterNumber)
                self.clusters[clusterNumber] = []   

            print("Current point that will be used", currentPoint)

        print()
        print("All points have been clustered.")
        for cluster in self.clusters.keys():
            print("Cluster ", cluster, " contains ", len(self.clusters[cluster]), " points.")


    #graph clusters in different colors
    def graphClusters(self):
        colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1)]
        clusterValues = [k for k in self.clusters.values()]
        count = 0
        for cluster in clusterValues:
            for point in cluster:
                self.graph.addPoint(point, colors[count])
            count += 1
            if count == len(colors):
                count = 0

        #graphing outliers
        for outlier in self.outliers:
            #graphing in grays
            self.graph.addOutlier(outlier,(0.5,0.5,0.5,1))

    def getARI(self):
        #find original classification for each point in self.points
        for point in self.originalPoints:
            self.clusteredLabels.append(self.pointsDict[point][1] if point in self.pointsDict else self.maxClusterNumber + 1)
        #convert to np array
        self.clusteredLabels = np.array(self.clusteredLabels)
        #find the adjusted rand index
        return adjusted_rand_score(self.originalLabels, self.clusteredLabels)
                

#skewed dataset generation. Source: https://machinelearningmastery.com/how-to-develop-an-intuition-skewed-class-distributions/ 
def get_dataset(proportions, centers, stdevs):
    # determine the number of classes
    n_classes = len(proportions)
    # determine the number of examples to generate for each class
    largest = max([v for k,v in proportions.items()])
    n_samples = largest * n_classes
    # create dataset
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2, random_state=1, cluster_std=stdevs)
    # collect the examples
    X_list, y_list = list(), list()
    for k,v in proportions.items():
        row_ix = np.where(y == k)[0]
        selected = row_ix[:v]
        X_list.append(X[selected, :])
        y_list.append(y[selected])
    return np.vstack(X_list), np.hstack(y_list)

if __name__ == '__main__':
    #getting k parameter from the user
    dNum = int(input("Which test dataset would you like to use? (1-6) "))
    default = input("Use default number of instances? (Y/N) ")
    d = True if default == "Y" else False
    numOfInstances = 0
    if d is False:
        numOfInstances = int(input("How many instances? "))
    k=int(input("How many nearest neighbors would you like to use? "))
    #determining if user wants to remove outliers
    RO = input("Remove outliers from dataset? (Y/N) ")
    RO = True if RO == "Y" else False
    #test datasets
    if d is True:
        data1 = datasets.make_moons(500,noise=0.05, random_state=0)[0].tolist() #works with k=10
        labels1 = datasets.make_moons(500,noise=0.05, random_state=0)[1]

        data2 = datasets.make_moons(500,noise=0.11, random_state=0)[0].tolist() #works with k = 5, removing outliers
        labels2 = datasets.make_moons(500,noise=0.11, random_state=0)[1]

        data3 = datasets.make_circles(500,noise=0.05, factor=0.5, random_state=0)[0].tolist() #works with k = 5
        labels3 = datasets.make_circles(500,noise=0.05, factor=0.5, random_state=0)[1]

        data4 = datasets.make_circles(500,noise=0.11, factor=0.5,  random_state=0)[0].tolist() #this is the dataset it fails on, as they are not well separated clusters. displayed test is with k=2, removing outliers
        labels4 = datasets.make_circles(500,noise=0.11, factor=0.5,  random_state=0)[1]

        stdevs = [0.5,3,0.5]
        proportions = {0:100, 1:500, 2:100}
        centers = [(-10,0),(0,0),(10,0)]
        data5 = get_dataset(proportions,centers,stdevs)[0].tolist() #works with k=5
        labels5 = get_dataset(proportions,centers,stdevs)[1]
        
        stdevs = [0.7,2.7,0.7]
        proportions = {0:200, 1:500, 2:200}
        centers = [(-7,0),(0,0),(7,0)]
        data6 = get_dataset(proportions,centers,stdevs)[0].tolist() #works with k=3, removing outliers
        labels6 = get_dataset(proportions,centers,stdevs)[1]
    else:
        data1 = datasets.make_moons(numOfInstances,noise=0.05, random_state=0)[0].tolist()
        labels1 = datasets.make_moons(numOfInstances,noise=0.05, random_state=0)[1]

        data2 = datasets.make_moons(numOfInstances,noise=0.11, random_state=0)[0].tolist()
        labels2 = datasets.make_moons(numOfInstances,noise=0.11, random_state=0)[1]

        data3 = datasets.make_circles(numOfInstances,noise=0.05, factor=0.5, random_state=0)[0].tolist() 
        labels3 = datasets.make_circles(numOfInstances,noise=0.05, factor=0.5, random_state=0)[1]

        data4 = datasets.make_circles(numOfInstances,noise=0.11, factor=0.5,  random_state=0)[0].tolist() 
        labels4 = datasets.make_circles(numOfInstances,noise=0.11, factor=0.5,  random_state=0)[1]

        stdevs = [0.5,3,0.5]
        n1 = int(numOfInstances/7.0)
        n2 = int(5*(numOfInstances/7.0))
        proportions = {0:n1, 1:n2, 2:n1}
        centers = [(-10,0),(0,0),(10,0)]
        data5 = get_dataset(proportions,centers,stdevs)[0].tolist() 
        labels5 = get_dataset(proportions,centers,stdevs)[1]
        
        stdevs = [0.7,2.7,0.7]
        n1 = int(2*(numOfInstances/9.0))
        n2 = int(5*(numOfInstances/9.0))
        proportions = {0:n1, 1:n2, 2:n1}
        centers = [(-7,0),(0,0),(7,0)]
        data6 = get_dataset(proportions,centers,stdevs)[0].tolist()
        labels6 = get_dataset(proportions,centers,stdevs)[1]

    dataLists = [data1, data2, data3, data4, data5, data6]
    labelLists = [labels1, labels2, labels3, labels4, labels5, labels6]

    #generating dataset
    list = dataLists[dNum-1]
    list = [tuple(l) for l in list] #converting to tuples
    label = labelLists[dNum-1]

    #initializing algorithm object with dataset
    knntc = KNNTC(list, label)

    #saveing the graph of the initial unclustered points
    for point in knntc.points:
        knntc.graph.addPoint(point, (1,0,0,1))
    #knntc.graph.save("initialPoints.png")

    #if the user said to remove outliers
    if RO:
        knntc.findDistances()
        knntc.removeOutliers()

    #making initial clusters from pre-processed point list
    knntc.cluster(k)

    #get the adjusted rand index
    printNames = ["Moons 1", "Moons 2", "Circles 1", "Circles 2", "Densities 1", "Densities 2"]
    print("ARI for KNN-TC " + str(printNames[dNum-1]) + " is " + str(knntc.getARI()))

    #graph clusters in different colors
    knntc.graph.clearGraph()
    knntc.graphClusters()
    names = ["moons1", "moons2", "circles1", "circles2", "densities1", "densities2"]
    knntc.graph.save("1_knntc_" + str(names[dNum-1]) + ".png")
    knntc.graph.show()