import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial import cKDTree

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes, M=1e2):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = np.zeros((n_nodes, n_nodes))

        tril_mask = np.tri(n_nodes, n_nodes, k=-1, dtype=bool)

        X_centered = [X[kmeans.labels_ == i] - kmeans.cluster_centers_[i] for i in range(n_nodes)]

        counts = np.array([X_centered[i].shape[0] for i in range(n_nodes)])
        counts = (counts[:, np.newaxis] + counts[np.newaxis, :])[tril_mask]
        
        segments = kmeans.cluster_centers_[:, np.newaxis] - kmeans.cluster_centers_[np.newaxis, :]
        dists = np.einsum('ij,ij->i', segments[tril_mask], segments[tril_mask])

        for i in range(1, n_nodes):
            projs_i = np.dot(X_centered[i], segments[i, :i].T)
            affinity[i, :i] = np.maximum(projs_i, 0).sum(axis=0)
            for j in range(i):
                projs_j = np.dot(X_centered[j], segments[j, i])
                affinity[i, j] += np.maximum(projs_j, 0).sum()

        affinity[tril_mask] = np.power(affinity[tril_mask] / (counts * dists), .5)
        affinity += affinity.T

        q1 = np.quantile(affinity, .25)
        q3 = np.quantile(affinity, .75)

        gamma = np.log(M) / (q3 - q1)
        affinity = np.exp(gamma * affinity)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed', assign_labels='cluster_qr').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        min_dists = np.array([cKDTree(cluster).query(x, 1)[0] for cluster in self.clusters])
        return min_dists.argmin(axis=0) 
