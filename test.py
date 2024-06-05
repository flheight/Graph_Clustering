import numpy as np
import matplotlib.pyplot as plt
from graph import Graph

from sklearn.metrics import adjusted_rand_score

data = np.genfromtxt('impossible.csv', delimiter=',')

X, y = data[:, :2], data[:, 2]

plt.scatter(X[:, 0], X[:, 1], color='black', s=.5)

net = Graph(n_classes=7)

net.fit(X, 200)

colors = plt.cm.tab10(np.arange(7))

guess = net.predict(X)

for i in range(7):
    plt.scatter(net.clusters[i][:, 0], net.clusters[i][:, 1], color=colors[i], label=f'Cluster {i}')

plt.show()


ari = adjusted_rand_score(y, guess)
print(f"Adjusted Rand Index: {ari}")
