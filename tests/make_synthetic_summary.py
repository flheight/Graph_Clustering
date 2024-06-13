import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

#set seed
np.random.seed(0)

# Function to load data
def load_data(file_name):
    data = np.genfromtxt('datasets/circles.csv', delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

# List of files
files = ['impossible.csv', 'moons.csv', 'circles.csv', 'smile.csv']

# Number of repetitions
N = 10

# Initialize results dictionary
results = {'DBSCAN': {}, 'KMeans': {}, 'GaussianMixture': {}, 'AgglomerativeClustering': {}}

# Iterate through each file
for file in files:
    # Load data
    X, y = load_data(file)

    # Initialize arrays for each algorithm and metric
    dbscan_ari = np.zeros(N)
    dbscan_nmi = np.zeros(N)
    kmeans_ari = np.zeros(N)
    kmeans_nmi = np.zeros(N)
    gm_ari = np.zeros(N)
    gm_nmi = np.zeros(N)
    agg_ari = np.zeros(N)
    agg_nmi = np.zeros(N)

    for i in range(N):
        # DBSCAN
        dbscan = DBSCAN()
        y_pred_dbscan = dbscan.fit_predict(X)
        dbscan_ari[i] = adjusted_rand_score(y, y_pred_dbscan)
        dbscan_nmi[i] = normalized_mutual_info_score(y, y_pred_dbscan)

        # KMeans
        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=i)
        y_pred_kmeans = kmeans.fit_predict(X)
        kmeans_ari[i] = adjusted_rand_score(y, y_pred_kmeans)
        kmeans_nmi[i] = normalized_mutual_info_score(y, y_pred_kmeans)

        # Gaussian Mixture
        gm = GaussianMixture(n_components=len(np.unique(y)), random_state=i)
        y_pred_gm = gm.fit_predict(X)
        gm_ari[i] = adjusted_rand_score(y, y_pred_gm)
        gm_nmi[i] = normalized_mutual_info_score(y, y_pred_gm)

        # Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=len(np.unique(y)))
        y_pred_agg = agg_clustering.fit_predict(X)
        agg_ari[i] = adjusted_rand_score(y, y_pred_agg)
        agg_nmi[i] = normalized_mutual_info_score(y, y_pred_agg)

    # Store results in dictionary
    results['DBSCAN'][file] = {
        'ARI': dbscan_ari,
        'NMI': dbscan_nmi
    }

    results['KMeans'][file] = {
        'ARI': kmeans_ari,
        'NMI': kmeans_nmi
    }

    results['GaussianMixture'][file] = {
        'ARI': gm_ari,
        'NMI': gm_nmi
    }

    results['AgglomerativeClustering'][file] = {
        'ARI': agg_ari,
        'NMI': agg_nmi
    }

print(results)
