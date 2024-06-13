import numpy as np
from sklearn.cluster import kmeans_plusplus
from scipy.sparse.csgraph import laplacian
import faiss

class _KMeans:
    def __init__(self, n_clusters, n_iter=20, random_state=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X):
        index = faiss.IndexFlatL2(X.shape[1]) 
        kmeans = faiss.Clustering(X.shape[1], self.n_clusters)
        init_centroids = kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.random_state)[0].astype(np.float32)

        kmeans.centroids.resize(init_centroids.size)
        faiss.memcpy(kmeans.centroids.data(), faiss.swig_ptr(init_centroids), init_centroids.size * 4)
        kmeans.niter = self.n_iter 
        kmeans.min_points_per_centroid = 0
        kmeans.max_points_per_centroid = -1
        kmeans.train(X.astype(np.float32), index)

        self.cluster_centers_ = faiss.vector_to_array(kmeans.centroids).reshape(self.n_clusters, X.shape[1])
        self.labels_ = index.search(X.astype(np.float32), 1)[1].ravel()

class _SpectralClustering:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, affinity):
        L = laplacian(affinity, normed=True)

        eigvecs = np.linalg.eigh(L)[1]
        eigvecs = eigvecs[:, :self.n_clusters]
        eigvecs /= np.linalg.norm(eigvecs, axis=1)[:, np.newaxis]
        kmeans = _KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(eigvecs)

        self.labels_ = kmeans.labels_

class SpectralBridges:
    def __init__(self, n_clusters, n_nodes, random_state=None):
        self.n_clusters = n_clusters
        self.n_nodes = n_nodes
        self.random_state = random_state

    def fit(self, X, M=1e4):
        kmeans = _KMeans(n_clusters=self.n_nodes, random_state=self.random_state)
        kmeans.fit(X)

        affinity = np.empty((self.n_nodes, self.n_nodes))

        X_centered = [X[kmeans.labels_ == i] - kmeans.cluster_centers_[i] for i in range(self.n_nodes)]

        counts = np.array([X_centered[i].shape[0] for i in range(self.n_nodes)])
        counts = counts[np.newaxis, :] + counts[:, np.newaxis]

        segments = kmeans.cluster_centers_[np.newaxis, :] - kmeans.cluster_centers_[:, np.newaxis]
        dists = np.einsum('ijk,ijk->ij', segments, segments)
        np.fill_diagonal(dists, 1)

        for i in range(self.n_nodes):
            projs = np.dot(X_centered[i], segments[i].T)
            affinity[i] = np.maximum(projs, 0).sum(axis=0)

        affinity = np.power((affinity + affinity.T) / (counts * dists), .5)
        affinity -= .5 * affinity.max()

        q1, q3 = np.quantile(affinity, [.25, .75])

        gamma = np.log(M) / (q3 - q1)
        affinity = np.exp(gamma * affinity)

        spclus = _SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state)
        spclus.fit(affinity)

        self.clusters = [kmeans.cluster_centers_[spclus.labels_ == i] for i in range(self.n_clusters)]

    def predict(self, x):
        min_dists = np.empty((self.n_clusters, x.shape[0]))

        for i, cluster in enumerate(self.clusters):
            index = faiss.IndexFlatL2(x.shape[1])
            index.add(cluster.astype(np.float32))
            min_dists[i] = index.search(x.astype(np.float32), 1)[0].ravel()

        return min_dists.argmin(axis=0)
