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
    data = np.genfromtxt(f'datasets/{file_name}', delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

# List of files
files = ['impossible.csv', 'moons.csv', 'circles.csv', 'smile.csv']

# Number of repetitions
N = 200

# Initialize results dictionary
results = {'DBSCAN': {}, 'KMeans': {}, 'GaussianMixture': {}, 'AgglomerativeClustering': {}}

# Iterate through each file
for file in files:
    # Load data
    X, y = load_data(file)

    # Number of classes
    n_clusters = np.unique(y).shape[0]

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
        kmeans = KMeans(n_clusters=n_clusters, random_state=i)
        y_pred_kmeans = kmeans.fit_predict(X)
        kmeans_ari[i] = adjusted_rand_score(y, y_pred_kmeans)
        kmeans_nmi[i] = normalized_mutual_info_score(y, y_pred_kmeans)

        # Gaussian Mixture
        gm = GaussianMixture(n_components=n_clusters, random_state=i)
        y_pred_gm = gm.fit_predict(X)
        gm_ari[i] = adjusted_rand_score(y, y_pred_gm)
        gm_nmi[i] = normalized_mutual_info_score(y, y_pred_gm)

        # Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred_agg = agg_clustering.fit_predict(X)
        agg_ari[i] = adjusted_rand_score(y, y_pred_agg)
        agg_nmi[i] = normalized_mutual_info_score(y, y_pred_agg)

        print(f"{int((100 * i) / (len(files) * N))}% done")

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

# Define datasets, methods, and metrics
dataset_labels = ["Impossible", "Moons", "Circles", "Smile"]
methods = ["DBSCAN", "KMeans", "GaussianMixture", "AgglomerativeClustering"]
metrics = ["ARI", "NMI"]

# Define colors for each method
method_colors = {
    "DBSCAN": "orange",
    "KMeans": "blue",
    "GaussianMixture": "green",
    "AgglomerativeClustering": "red",
    "SpectralBridges": "purple"
}

# Define method labels for the legend
method_labels = {
    "DBSCAN": "DB",
    "KMeans": "KM",
    "GaussianMixture": "EM",
    "AgglomerativeClustering": "WC",
    "SpectralBridges": "SB"
}

# Create boxplots for ARI and NMI
fig = make_subplots(
    rows=len(files),  # Number of rows equals the number of datasets
    cols=2,  # Two columns for ARI and NMI
    shared_xaxes=True,
    subplot_titles=("ARI", "NMI"),
    vertical_spacing=0.05,  # Reduce vertical spacing
    horizontal_spacing=0.075  # Reduce horizontal spacing
)

# Helper function to add traces
def add_traces(fig, metric, col):
    for dataset in files:
        for method in methods:
            data = results[method][dataset][metric]
            if data.size > 0:
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=method_labels[method],  # Use method labels for legend
                        boxmean='sd',
                        marker=dict(color=method_colors[method]),  # Assign color based on method
                        text=[method_labels[method]] * len(data),  # Text annotation for each box
                        hoverinfo="text",
                    ),
                    row=files.index(dataset) + 1, col=col  # Row index starts from 1
                )

# Add traces for ARI and NMI
for col, metric in enumerate(metrics, start=1):
    add_traces(fig, metric, col)

# Update layout
for i, dataset in enumerate(dataset_labels, start=1):
    fig.update_yaxes(title_text=dataset, row=i, col=1)
    fig.update_xaxes(title_text="", row=i, col=1)  # Remove x-axis label for better space utilization

fig.update_layout(
    height=1600,
    width=1200,
    title_text="",
    showlegend=False,
    legend_title_text="Methods",
    font=dict(size=20)  # Increase font size
)

for i in range(2):
    fig.layout.annotations[i].font.size = 25  # Increase subplot titles font size


# Save the figure as an HTML file
pio.write_image(fig, "synthetic_summary.pdf")

