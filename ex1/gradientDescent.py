from computeCost import computeCost
import numpy as np


def gradientDescent(X, y, theta, alpha, num_iters):
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

        # ============================================================

        h = np.dot(X, theta)
        
        theta = theta - ((alpha / m) * (np.dot(X.T,(h - y)))) 
        
        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history


# a = np.matrix('1 5; 1 2; 1 4; 1 5')
# b = np.matrix('1;6;4;2')
# c = np.matrix('0.0;0.0')

# (l,m) = gradientDescent(a, b, c, 0.01, 1000)
# print l
# print m