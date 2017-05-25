from sigmoid import sigmoid
from numpy import squeeze, asarray

from numpy import log, matmul, ones, asmatrix
from sigmoid import sigmoid
import numpy as np

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = asmatrix(data[:, 0:2])
y = (asmatrix(data[:, 2])).T
(m, n) = X.shape
X = np.append(ones((m, 1)), X, axis=1)
theta = asmatrix(np.zeros(n + 1))


def gradientFunction(theta, X, y):
	"""
	Compute cost and gradient for logistic regression with regularization

	computes the cost of using theta as the parameter for regularized logistic regression and the
	gradient of the cost w.r.t. to the parameters.
	"""

	m = len(y)   # number of training examples
	grad = 0
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the gradient of a particular choice of theta.
	#               Compute the partial derivatives and set grad to the partial
	#               derivatives of the cost w.r.t. each parameter in theta
	h = sigmoid(np.dot( X,theta.T))
	grad = (1. / m) * np.dot((X.T) , (h - y))
	# =============================================================

	return grad.T


# print (gradientFunction(theta, X, y))