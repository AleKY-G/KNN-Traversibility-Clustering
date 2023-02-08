import numpy as np
import math
import sys
import copy
import decimal
import random
from sklearn import datasets  
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import time

clusterTimes = []
dictTimes = []

class Graph:
    def __init__(self):
        self.points = []
        self.colors = []
        self.outlierColors = []
        self.outliers = []

    def addPoint(self, point, color):
        self.points.append(point)
        self.colors.append(color)

    def addOutlier(self, point, color):
        self.outliers.append(point)
        self.outlierColors.append(color)

    def clearGraph(self):
        self.points = []
        self.colors = []
        self.outlierColors = []
        self.outliers = []
        
    def save(self, name):
        #import matplotlib.pyplot as plt
        plt.clf()
        plt.scatter(*zip(*self.points), c=self.colors)
        if len(self.outliers) > 0:
            plt.scatter(*zip(*self.outliers), c=self.outlierColors)
        plt.savefig(name)

    def show(self):
        #import matplotlib.pyplot as plt
        plt.clf()
        plt.scatter(*zip(*self.points), c=self.colors)
        #if len(self.outliers) > 0:
        #    plt.scatter(*zip(*self.outliers), c=self.outlierColors)
        plt.plot(*zip(*self.points),color='green', linestyle='dashed', linewidth = 1,)
        plt.show()

class Algorithm:
    def __init__(self, points):
        self.points = points
        self.distancesDict = {}
        self.distances = []
        self.outliers = []
        self.clusters = {}
        self.pointsDict = {}
        self.graph = Graph()
        self.dictTimes = []
        self.clusterTimes = []

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


    #points is a list - need to remove the outliers
    def removeOutliers(self, cutoff=1.5, both=False):
        #find outliers in distances
        outliers = []
        self.distances.sort()
        q1 = math.ceil(0.25 * len(self.distances))
        q3 = math.ceil(0.75 * len(self.distances))
        iqr = self.distances[q3] - self.distances[q1]

        #no upper bound because we only care about points that are too far away
        #set lowerbound to lowest possible integer
        lowerBound = -sys.maxsize
        if both==True:
            print("using both")
            lowerBound = self.distances[q1] - (cutoff * iqr)
        upperBound = self.distances[q3] + (cutoff * iqr)

        for i in range(len(self.distances)):
            if self.distances[i] > upperBound:
                outliers.append(self.distances[i])
            elif self.distances[i] < lowerBound:
                print("lower bound outlier")
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

    def distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def neighborsOrdered(self, p, points):
        #order neighbors by distance from p. If there are multiple points with the same distance, order them by leftmost point, and if they share the same x, then order them by lowest point
        neighbors = []
        for p2 in points:
            if p2 != p:
                neighbors.append(p2)
        neighbors.sort(key=lambda point: (self.distance(p, point), point[0], point[1]))
        #points.sort(key=lambda point: (sqrt(point[0]**2 + point[1]**2), point[0]))

        return neighbors

    #making clusters from pre-processed point list
    def cluster(self, k, startingPoint = None):
        global clusterTimes
        global dictTimes
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
        listlen = len(self.points)
        start = time.time()
        self.pointsCopy = copy.deepcopy(self.points)
        for point in self.pointsCopy:
            neighbors = self.neighborsOrdered(point, self.pointsCopy)
            self.pointsDict[point] = [neighbors, None]
        end = time.time()
        print("Time to populate pointsDict:", end-start)
        dictTimes.append((listlen,end-start))

        currentPoint = startingPoint
        clusterNumber = 0
        maxClusterNumber = 0
        self.clusters[clusterNumber] = []

        start = time.time()
        totalLen = len(self.pointsCopy)
        clustered = 0
        while clustered < totalLen: 
            #remove the current point from the available points and add it to the current cluster
            #self.pointsCopy.remove(currentPoint)
            self.clusters[clusterNumber].append(currentPoint)
            self.pointsDict[currentPoint][1] = clusterNumber
            clustered += 1
            #print("Added " + str(currentPoint) + " to cluster " + str(clusterNumber))

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
                #print("All of the neighbors of ", currentPoint, " are already in a cluster. Searching for a new current point...")
                #iterate through all of the currentPoint's neighbors
                for currPointNeighbor in self.pointsDict[currentPoint][0]:
                    #if currPointNeighbor has a nearest neighbor that is assigned to a cluster and currPointNeighbor is not assigned to a cluster:
                    if self.pointsDict[currPointNeighbor][1] is None:
                        for j in range(k):
                            if self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1] is not None and flag1 == True:
                                currentPoint = currPointNeighbor
                                #print("New current point is ", currentPoint, " and cluster number is ", self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1], " because of ",self.pointsDict[currPointNeighbor][0][j], "")
                                clusterNumber = self.pointsDict[self.pointsDict[currPointNeighbor][0][j]][1]
                                flag1 = False

            #if currentPoint STILL remains unchanged, that means that there were no unclustered points that had a k nearest neighbor that was already in a cluster
            #so we just pick the first unclustered neighbor, and start a new cluster
            if currentPoint == prevPoint or clusterNumber is None:
                #print("All neighbors (that are not the k nearest neighbors of) " + str(currentPoint) + "are unclustered. Choosing a new current point...")
                newPoint = None
                j = k
                while newPoint is None:
                    newPoint = self.pointsDict[currentPoint][0][j] if self.pointsDict[self.pointsDict[currentPoint][0][j]][1] is None else None
                    j+=1 
                currentPoint = newPoint
                maxClusterNumber += 1
                clusterNumber = maxClusterNumber
                #print("New cluster point is ", currentPoint, " and cluster number is ", clusterNumber)
                self.clusters[clusterNumber] = []   

            #print("Current point that will be used", currentPoint)

            #if currentPoint STILL remains unchanged, that means that none of the remaining unclustered neighbors have a neighbor that is assigned to a cluster
        #print()
        #print("All points have been clustered." + str(self.clusters.keys()) + "")
        for cluster in self.clusters.keys():
            print("Cluster ", cluster, " contains ", len(self.clusters[cluster]), " points.")
        end = time.time()
        print("Time to cluster:", end-start)
        clusterTimes.append((listlen,end-start))
        print()


    #graph clusters in different colors
    def graphClusters(self):
        colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),   (1,1,0,1),(1,0,1,1),(0,1,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),]
        #self.clusters = {"0":[(randint(0, 9), randint(0, 9)) for i in range(5)],"1":[(randint(0, 9), randint(0, 9)) for i in range(5)],"2":[(randint(0, 9), randint(0, 9)) for i in range(5)]}
        clusterValues = [k for k in self.clusters.values()]
        count = 0
        for cluster in clusterValues:
            for point in cluster:
                self.graph.addPoint(point, colors[count])
            count += 1

        #graphing outliers
        for outlier in self.outliers:
            #graphing in grays
            self.graph.addOutlier(outlier,(0.5,0.5,0.5,1))


def generateRand(l, u):
    return float(decimal.Decimal(random.randrange(l*100, u*100))/100)


#skewed dataset generation. Source: https://machinelearningmastery.com/how-to-develop-an-intuition-skewed-class-distributions/ 
def get_dataset(proportions, centers,stdevs):
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

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
    plt.clf()
    # create scatter plot for samples from each class
    n_classes = len(np.unique(y))
    for class_value in range(n_classes):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)[0]
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
    # show a legend
    plt.legend()
    plt.savefig("skewedDataOriginal.png")


if __name__ == '__main__':
    r = int(input("Up to how many points would you like to test? "))
    i = int(input("How many increments would you like to test? "))
    k = int(input("What value of k to use for each test? "))
    sizes = [int(j * (r/i)) for j in range(1,i+1)]
    print("Point amounts to be tested: ", sizes)
    plt.clf()
    for s in sizes:
        list =  datasets.make_circles(s,noise=0.05,factor=0.3,random_state=0)[0].tolist()
        list = [tuple(l) for l in list]

        #where points are going to be plotted
        algorithm = Algorithm(list)

        #making initial clusters from pre-processed point list
        print("Points size is ", s)
        algorithm.cluster(k)
        
        #algorithm.graph.clearGraph()
    plt.plot(*zip(*dictTimes), label="Dictionary Generation Times", color="red", linewidth=2)
    plt.plot(*zip(*clusterTimes), label="Clustering Times", color="blue", linewidth=2)
    plt.legend()
    plt.xlabel("Number of Points")
    plt.ylabel("Time (seconds)")
    plt.savefig("knntc_timed.png")
    plt.show()