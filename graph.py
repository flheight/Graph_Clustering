import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes, M=1e2):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = np.zeros((n_nodes, n_nodes))

        segments = kmeans.cluster_centers_[:, np.newaxis] - kmeans.cluster_centers_[np.newaxis, :]
        norms = np.linalg.norm(segments, axis=2)

        for i in range(n_nodes):
            for j in range(i):
                projs_i = np.dot(X[kmeans.labels_ == i] - kmeans.cluster_centers_[i], segments[i, j])
                score_i = np.maximum(projs_i, 0).sum()
                projs_j = np.dot(X[kmeans.labels_ == j] - kmeans.cluster_centers_[j], segments[j, i])
                score_j = np.maximum(projs_j, 0).sum()

                affinity[i, j] = np.power((score_i + score_j) / (projs_i.shape[0] + projs_j.shape[0]), .5) / norms[i, j]

        affinity += affinity.T

        q1 = np.quantile(affinity, .25)
        q3 = np.quantile(affinity, .75)

        gamma = np.log(M) / (q3 - q1)
        affinity = np.exp(gamma * affinity)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
