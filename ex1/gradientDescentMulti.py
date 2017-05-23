from computeCostMulti import computeCostMulti
import numpy as np


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        h = np.dot(X, theta)
        theta = theta - ((alpha / m) * (np.dot(X.T, (h - y))))
        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


# X = np.matrix('2 1 3; 7 1 9; 1 8 1; 3 7 4 ')
# y = np.matrix('2 ; 5 ; 5 ; 6')
# theta = np.matrix('0;0;0')

# print gradientDescentMulti(X, y, theta, 0.01, 100)
