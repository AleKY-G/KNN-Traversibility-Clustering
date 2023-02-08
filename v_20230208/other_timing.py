import matplotlib.pyplot as plt
from sklearn.datasets import  make_moons
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import time

#timing dbscan, snn, single linkage, and kmeans on moons 1 repeatedly up to 20000 points
ptNums = [i*1000 for i in range(1,21)]
dbscan_times = []
snn_times = []
singlelinkage_times = []
kmeans_times = []

plt.clf()
for pt in ptNums:
    X, y = make_moons(pt,noise=0.05,random_state=0)
    
    #DBSCAN
    start = time.time()
    dbscan=DBSCAN(eps=0.105, min_samples=4) 
    pred_labels = dbscan.fit_predict(X)
    end = time.time() - start
    dbscan_times.append((pt, end))

    #SNN
    k = 10
    threshold = 4
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    knn_graph = nbrs.kneighbors_graph(X)
    similarity_matrix = knn_graph.multiply(knn_graph)
    similarity_matrix.data = np.ones_like(similarity_matrix.data)
    dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
    pred_labels = dbscan.fit_predict(similarity_matrix)
    end = time.time() - start
    snn_times.append((pt, end))

    #SINGLE LINKAGE
    start = time.time()
    ac = AgglomerativeClustering(n_clusters = 2,linkage='single')
    pred_labels = ac.fit_predict(X)
    end = time.time() - start
    singlelinkage_times.append((pt, end))

    #KMEANS
    start = time.time()
    kmeans = KMeans(n_clusters=2, random_state=0)
    pred_labels = kmeans.fit_predict(X)
    end = time.time() - start
    kmeans_times.append((pt, end))


plt.plot(*zip(*dbscan_times), label="DBSCAN Clustering Times", color="red", linewidth=2)
plt.plot(*zip(*snn_times), label="SNN Clustering Times", color="blue", linewidth=2)
plt.plot(*zip(*singlelinkage_times), label="Single Linkage Clustering Times", color="green", linewidth=2)
plt.plot(*zip(*kmeans_times), label="KMeans Clustering Times", color="gray", linewidth=2)
plt.legend()
plt.xlabel("Number of Points")
plt.ylabel("Time (seconds)")
plt.show()
