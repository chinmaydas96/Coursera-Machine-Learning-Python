from scipy.optimize import minimize

import numpy as np

from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, Lambda, method='CG', maxiter=200):

    """trains linear regression using
    the dataset (X, y) and regularization parameter lambda. Returns the
    trained parameters theta.
    """

# Initialize Theta
    initial_theta = np.zeros(X.shape[1])

# Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, Lambda)[0]
    gradFunction = lambda t: linearRegCostFunction(X, y, t, Lambda)[1]

    result = minimize(costFunction, initial_theta, method=method, jac=None, options={'disp': True, 'maxiter': maxiter})

    return result.x
