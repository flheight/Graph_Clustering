import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial import cKDTree

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes, M=1e2):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = np.zeros((n_nodes, n_nodes))

        X_centered = [X[kmeans.labels_ == i] - kmeans.cluster_centers_[i] for i in range(n_nodes)]
        segments = kmeans.cluster_centers_[:, np.newaxis] - kmeans.cluster_centers_[np.newaxis, :]
        dists = np.einsum('ijk,ijk->ij', segments, segments)
        np.fill_diagonal(dists, 1)

        for i in range(1, n_nodes):
            projs_i = np.dot(X_centered[i], segments[i, :i].T)
            scores_i = np.maximum(projs_i, 0).sum(axis=0)
            for j in range(i):
                projs_j = np.dot(X_centered[j], segments[j, i])
                score_j = np.maximum(projs_j, 0).sum()
                affinity[i, j] = (scores_i[j] + score_j) / (X_centered[i].shape[0] + X_centered[j].shape[0])

        affinity = np.power((affinity + affinity.T) / dists, .5)

        q1 = np.quantile(affinity, .25)
        q3 = np.quantile(affinity, .75)

        gamma = np.log(M) / (q3 - q1)
        affinity = np.exp(gamma * affinity)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed', assign_labels='cluster_qr').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        min_dists = np.array([cKDTree(cluster).query(x, 1)[0] for cluster in self.clusters])
        return min_dists.argmin(axis=0) 
