from computeCost import computeCost
import numpy as np
import pandas as pd
df = pd.read_csv('ex1data1.txt', names=['X', 'y'])

y = df['y']
m = y.size


X = df.as_matrix(columns=['X'])
X = np.append(np.ones((m, 1)), X, axis=1)
theta = np.zeros(2)
alpha = 0.01
num_iters = 1500


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
        h = np.dot(X, theta)
        theta = theta - (alpha / m) * np.dot(X.T, (h - y))

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

gradientDescent(X, y, theta, alpha, num_iters)
