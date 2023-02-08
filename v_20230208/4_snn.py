import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

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

#ARI for Moons 1 (0.05)
X, y = make_moons(500,noise=0.05,random_state=0)
k = 10
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Moons 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_moons1.png")

#ARI for Moons 2 (0.11)
X, y = make_moons(500,noise=0.11,random_state=0)
k = 4
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Moons 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_moons2.png")

#ARI for Circles 1 (0.05)
X, y = make_circles(500,noise=0.05,factor=0.5,random_state=0)
k = 12
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Circles 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_circles1.png")

#ARI for Circles 2 (0.11)
X, y = make_circles(500,noise=0.11,factor=0.5,random_state=0)
k = 3
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Circles 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_circles2.png")

#ARI for Densities 1
stdevs = [0.5,3,0.5]
proportions = {0:100, 1:500, 2:100}
centers = [(-10,0),(0,0),(10,0)]
X, y = get_dataset(proportions,centers,stdevs)
k = 4
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Densities 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_densities1.png")

#ARI for Densities 2
stdevs = [0.7,2.7,0.7]
proportions = {0:200, 1:500, 2:200}
centers = [(-7,0),(0,0),(7,0)]
X, y = get_dataset(proportions,centers,stdevs)
k = 4
threshold = 4
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
knn_graph = nbrs.kneighbors_graph(X)
similarity_matrix = knn_graph.multiply(knn_graph)
similarity_matrix.data = np.ones_like(similarity_matrix.data)
dbscan = DBSCAN(eps=threshold,min_samples=k, metric='precomputed')
pred_labels = dbscan.fit_predict(similarity_matrix)
ari = adjusted_rand_score(y, pred_labels)
print("ARI for SNN Densities 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=pred_labels,cmap='rainbow')
plt.savefig("4_snn_densities2.png")