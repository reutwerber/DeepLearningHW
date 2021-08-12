# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd

def kmeans_example():
    #KMeans Example
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50);

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.savefig('kmeans_example.png')


def plot_basic_figure():
    # Create the figure and two axes (two rows, one column)
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Create a plot of y = sin(x) on the first row
    x1 = np.linspace(0, 4 * np.pi, 100)
    y1 = np.sin(x1)
    ax1.plot(x1, y1)

    # Create a plot of y = cos(x) on the second row
    x2 = np.linspace(0, 4 * np.pi, 100)
    y2 = np.cos(x2)
    ax2.plot(x2, y2)

    # Save the figure
    plt.savefig('sin_cos.png')
