import matplotlib.pyplot as plt
import numpy as np
from math import isinf
from multivariateGaussian import multivariateGaussian

from show import show


def visualizeFit(X, mu, sigma2):
    """
    This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    n = np.linspace(0,35,71)
    X1 = np.meshgrid(n,n)
    Z = multivariateGaussian(np.column_stack((X1[0].T.flatten(), X1[1].T.flatten())),mu,sigma2)
    Z = Z.reshape(X1[0].shape)

    plt.plot(X[:, 0], X[:, 1],'bx')
    # Do not plot if there are infinities
    if not isinf(np.sum(Z)):
        plt.contour(X1[0], X1[1], Z, 10.0**np.arange(-20, 0, 3).T)
        show()
