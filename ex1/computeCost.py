import numpy as np
from numpy import ones

# import pandas as pd 
# df = pd.read_csv('ex1data1.txt',names=['X','y'])

# y = df['y']
# m = y.size


# X = df.as_matrix(columns=['X'])
# X = np.append(ones((m,1)),X,axis=1)
# theta = np.zeros(2)

def computeCost(X, y, theta):
	"""
	   computes the cost of using theta as the parameter for linear 
	   regression to fit the data points in X and y
	"""
	m = y.size
	J = 0

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta
	#               You should set J to the cost.
	h = np.dot(X,theta)
	sq_error = np.sum(np.square(h - y))	
	J =  (sq_error) / (2 * m) 
# =========================================================================

	return J

# computeCost(X, y, theta)