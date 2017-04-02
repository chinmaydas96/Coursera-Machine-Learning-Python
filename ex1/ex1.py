from matplotlib import use, cm
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn import linear_model

from gradientDescent import gradientDescent
from computeCost import computeCost
from warmUpExercise import warmUpExercise
from plotData import plotData
from show import show

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print 'Running warmUpExercise ...'
print '5x5 Identity Matrix:'
warmup = warmUpExercise()
print warmup
raw_input("Program paused. Press Enter to continue...")

# ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = np.vstack(zip(np.ones(m),data[:,0]))
y = data[:, 1]

# Plot Data
# Note: You have to complete the code in plotData.py
print 'Plotting Data ...'
plotData(data)
show()

raw_input("Program paused. Press Enter to continue...")

# =================== Part 3: Gradient descent ===================
print 'Running Gradient Descent ...'
theta = np.zeros(2)

# compute and display initial cost
J = computeCost(X, y, theta)
print 'cost: %0.4f ' % J

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print 'Theta found by gradient descent: '
print '%s %s \n' % (theta[0], theta[1])

# Plot the linear fit
plt.figure()
plotData(data)
plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

raw_input("Program paused. Press Enter to continue...")

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print 'For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000)
print 'For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print 'Visualizing J(theta_0, theta_1) ...'

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, X.shape[0])
theta1_vals = np.linspace(-1, 4, X.shape[0])

# initialize J_vals to a matrix of 0's
J_vals=np.array(np.zeros(X.shape[0]).T)

for i in range(theta0_vals.size):
    col = []
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        col.append(computeCost(X, y, t.T))
    J_vals=np.column_stack((J_vals,col))

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals[:,1:].T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=8, cstride=8, alpha=0.3,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')
show()

raw_input("Program paused. Press Enter to continue...")

# Contour plot
plt.figure()

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.clabel(ax, inline=1, fontsize=10)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(0.0, 0.0, 'rx', linewidth=2, markersize=10)
show()

raw_input("Program paused. Press Enter to continue...")

# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(X, y)

print 'Theta found by scikit: '
print '%s %s \n' % (regr.coef_[0], regr.coef_[1])

predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print 'For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000)
print 'For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000)

plt.figure()
plotData(data)
plt.plot(X[:, 1],  X.dot(regr.coef_), '-', color='black', label='Linear regression wit scikit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

raw_input("Program paused. Press Enter to continue...")


