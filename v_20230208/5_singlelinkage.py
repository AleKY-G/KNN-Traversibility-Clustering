import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs, make_circles, make_moons

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
ac = AgglomerativeClustering(n_clusters = 2,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X))
print("ARI for Single Linkage Moons 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X),cmap='rainbow')
plt.savefig("5_singlelinkage_moons1.png")


#ARI for Moons 2 (0.11)
X, y = make_moons(500,noise=0.11,random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']
ac = AgglomerativeClustering(n_clusters = 2,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X_principal))
print("ARI for Single Linkage Moons 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X_principal),cmap='rainbow')
plt.savefig("5_singlelinkage_moons2.png")


#ARI for Circles 1 (0.05)
X, y = make_circles(500,noise=0.05,factor=0.5,random_state=0)
ac = AgglomerativeClustering(n_clusters = 2,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X))
print("ARI for Single Linkage Circles 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X),cmap='rainbow')
plt.savefig("5_singlelinkage_circles1.png")


#ARI for Circles 2 (0.11)
X, y = make_circles(500,noise=0.11,factor=0.5,random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']
ac = AgglomerativeClustering(n_clusters = 2,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X_principal))
print("ARI for Single Linkage Circles 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X_principal),cmap='rainbow')
plt.savefig("5_singlelinkage_circles2.png")


#ARI for Densities 1
stdevs = [0.5,3,0.5]
proportions = {0:100, 1:500, 2:100}
centers = [(-10,0),(0,0),(10,0)]
X, y = get_dataset(proportions,centers,stdevs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']
ac = AgglomerativeClustering(n_clusters = 3,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X_principal))
print("ARI for Single Linkage Densities 1 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X_principal),cmap='rainbow')
plt.savefig("5_singlelinkage_densities1.png")


#ARI for Densities 2
stdevs = [0.7,2.7,0.7]
proportions = {0:200, 1:500, 2:200}
centers = [(-7,0),(0,0),(7,0)]
X, y = get_dataset(proportions,centers,stdevs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']
ac = AgglomerativeClustering(n_clusters = 3,linkage='single')
ari = adjusted_rand_score(y, ac.fit_predict(X_principal))
print("ARI for Single Linkage Densities 2 is",ari)
plt.clf()
plt.scatter(X[:,0],X[:,1],c=ac.fit_predict(X_principal),cmap='rainbow')
plt.savefig("5_singlelinkage_densities2.png")
