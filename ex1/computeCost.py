import numpy as np

def computeCost(X, y, theta):
	"""
	   computes the cost of using theta as the parameter for linear 
	   regression to fit the data points in X and y
"""
	m = y.size


	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	#               You should set J to the cost.
	
	h = np.dot(X,theta)
	sq_error = np.sum(np.square(h-y))
	J = (sq_error)*((1.0) /(2*m))
# =========================================================================

	return J	


# a = np.matrix('1 2; 1 3; 1 4; 1 5')
# b  = np.matrix('7;6;5;4')
# c = np.matrix('0.1;0.2')

# print (computeCost(a,b,c))
