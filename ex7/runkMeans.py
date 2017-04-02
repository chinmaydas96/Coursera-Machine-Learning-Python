from computeCentroids import computeCentroids
from plotProgresskMeans import plotProgresskMeans
from findClosestCentroids import findClosestCentroids
import matplotlib.pyplot as plt
import numpy as np
import itertools


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    """runs the K-Means algorithm on data matrix X, where each
    row of X is a single example. It uses initial_centroids used as the
    initial centroids. max_iters specifies the total number of interactions
    of K-Means to execute. plot_progress is a true/false flag that
    indicates if the function should also plot its progress as the
    learning happens. This is set to false by default. runkMeans returns
    centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Plot the data if we are plotting progress
    if plot_progress:
        plt.figure()

# Initialize values
    m, n = X.shape
    K = len(initial_centroids)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    c = itertools.cycle('012')
    rgb = np.eye(3)
# Run K-Means
    for i in range(max_iters):
    
        # Output progress
        print 'K-Means iteration %d/%d...' % (i, max_iters)

        # For each example in X, assign it to the closest centroid
        _, idx = findClosestCentroids(X, centroids)
    
        # Optionally, plot progress here
        if plot_progress:
            color = rgb[int(next(c))]
            plotProgresskMeans(X, np.array(centroids),
                               np.array(previous_centroids), idx, K, i, color)
            previous_centroids = centroids
            # raw_input("Press Enter to continue...")

    # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

# Hold off if we are plotting progress
    if plot_progress:
        pass
    # hold off
    return centroids, idx
