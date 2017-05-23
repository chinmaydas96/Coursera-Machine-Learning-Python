import numpy as np


def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = y.size
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

    h = np.dot(X, theta)
    sq_error = np.sum(np.square(h - y))
    J = (sq_error) * ((1.0) / (2 * m))
    # =========================================================================

    return J


# X = np.matrix(' 2 1 3; 7 1 9; 1 8 1; 3 7 4 ')
# y = np.matrix(' 2 ; 5 ; 5 ; 6')
# theta_test = np.matrix('0.4 ; 0.6 ; 0.8')
# print computeCostMulti(X, y, theta_test)
