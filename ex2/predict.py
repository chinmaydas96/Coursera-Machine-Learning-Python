from numpy import round
import numpy as np
from sigmoid import sigmoid


# theta = [-25.8142, 0.2111, 0.2070]
# theta = np.asmatrix(theta)
# X = [[1, 45, 85],[1, 40 ,50]]
# X = np.asmatrix(X)
# print X.shape
# print theta.shape

def predict(theta, X):

	""" computes the predictions for X using a threshold at 0.5
	(i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
	"""
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: Complete the following code to make predictions using
	#               your learned logistic regression parameters.
	#               You should set p to a vector of 0's and 1's
	#
	(m,n) = X.shape
	p = np.zeros((m,1)) 
	p = sigmoid(np.dot(X , theta.T)) 
	# =========================================================================
	# print "hello"
	return [1 if x >= 0.5 else 0 for x in p]
	

# print (predict(theta,X))	



