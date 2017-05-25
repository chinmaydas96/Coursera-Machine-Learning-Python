from numpy import log, matmul, ones, asmatrix
from sigmoid import sigmoid
import numpy as np

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = asmatrix(data[:, 0:2])
y = (asmatrix(data[:, 2])).T
(m, n) = X.shape
X = np.append(ones((m, 1)), X, axis=1)
theta = asmatrix(np.zeros(n + 1))

def costFunction(theta, X, y):
	""" computes the cost of using theta as the
	parameter for logistic regression and the
	gradient of the cost w.r.t. to the parameters."""

	# Initialize some useful values
	m = y.size  # number of training examples

	
	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta.
	#               You should set J to the cost.
	#               Compute the partial derivatives and set grad to the partial
	#               derivatives of the cost w.r.t. each parameter in theta
	#
	h = sigmoid(np.dot( X,theta.T))
    
	a = np.multiply(y , np.log(h))
	b = np.multiply((1 - y) , log(1 - h))

	J = -1 *(1./m ) * ((a+b).sum())
	# J = (1 / m) * (np.sum(np.transpose(a + b)))
	# Note: grad should have the same dimensions as theta
    
	return J
	

# print costFunction(theta, X, y)
