import matplotlib.pyplot as plt
import numpy as np
from show import show
def plotData(X, y):
    """plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    """
    plt.figure()

# Find Indices of Positive and Negative Examples
    pos = np.where(y==1, True, False).flatten()
    neg = np.where(y==0, True, False).flatten()

# Plot Examples
    plt.plot(X[pos,0], X[pos, 1], 'k+', linewidth=1, markersize=7)
    plt.plot(X[neg,0], X[neg, 1], 'ko', color='y', markersize=7)
    show()

