import numpy as np
import pandas as pd

df = pd.read_csv('ex1data1.txt', names=['X', 'y'])
y = df['y']
m = y.size


X = df.as_matrix(columns=['X'])
X = np.append(np.ones((m, 1)), X, axis=1)


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------

    theta = np.matmul(
        (np.matmul(np.linalg.inv(np.matmul((X.T), X)), X.T)), y)

# -------------------------------------------------------------

    return theta

# ============================================================


print (normalEqn(X, y))
