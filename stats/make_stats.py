import plotly.graph_objects as go

# Define the dimensions and scores
dim = ['h=8', 'h=16', 'h=32', 'h=64', 'h=784 (full)']
k_means_scores = [0.36, 0.36, 0.37, 0.37, 0.36]
em_scores = [0.40, 0.53, 0.45, 0.52, 0.50]
ac_scores = [0.39, 0.45, 0.50, 0.43, 0.42]
bridge_scores = [0.61, 0.73, 0.76, 0.79, 0.59]

# Define colors for each method and their lighter shades
colors = {
    'k-means': ('blue', 'lightblue'),
    'EM': ('green', 'lightgreen'),
    'Agglomerative Clustering (Ward linkage)': ('red', 'lightcoral'),
    'Spectral-Bridges': ('purple', 'plum')
}

# Helper function to determine the color of each bar
def get_bar_color(score, max_score, method):
    return colors[method][0] if score == max_score else colors[method][1]

# Create the figure with bar charts for each method
fig = go.Figure()

# Methods and their corresponding scores
methods = ['k-means', 'EM', 'Agglomerative Clustering (Ward linkage)', 'Spectral-Bridges']
scores = [k_means_scores, em_scores, ac_scores, bridge_scores]

# Plot bars for each method
for method, score in zip(methods, scores):
    max_scores = [max(k_means_scores[i], em_scores[i], ac_scores[i], bridge_scores[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))]
    ))

# Update layout with larger and bold axis titles and tick labels
fig.update_layout(
    xaxis_title='h',
    yaxis_title='ARI Score',
    xaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    yaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    barmode='group',
    legend=dict(
        font=dict(
            size=20,
            weight='bold'
        ),
        title_font_size=24,
        itemsizing='constant'
    )
)

# Save the figure as an HTML file
fig.write_html("stats.html")
