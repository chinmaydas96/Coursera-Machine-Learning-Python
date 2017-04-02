## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function. 
#
from matplotlib import use, cm
use('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from show import show

print 'Finding closest centroids.'

# Load an example dataset that we will be using
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
val, idx = findClosestCentroids(X, initial_centroids)

print 'Closest centroids for the first 3 examples:'
print idx[0:3].tolist()
print '(the closest centroids should be 0, 2, 1 respectively)'

raw_input("Program paused. Press Enter to continue...") 

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print 'Computing centroids means.'

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print 'Centroids computed after initial finding of closest centroids:'
for c in centroids:
    print c

print '(the centroids should be'
print '   [ 2.428301 3.157924 ]'
print '   [ 5.813503 2.633656 ]'
print '   [ 7.119387 3.616684 ]'

raw_input("Program paused. Press Enter to continue...") 


## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print 'Running K-Means clustering on example dataset.'

# Load an example dataset
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = [[3, 3], [6, 2], [8, 5]]

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print 'K-Means Done.'

raw_input("Program paused. Press Enter to continue...") 

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

print 'Running K-Means clustering on pixels from an image.'

#  Load an image of a bird
A = scipy.misc.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16 
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

raw_input("Program paused. Press Enter to continue...") 


## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print 'Applying K-Means to compress an image.'

# Find closest cluster members
_, idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
X_recovered = np.array([centroids[e] for e in idx])

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image 
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Original')
show()

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title('Compressed, with %d colors.' % K)
show()

raw_input("Program paused. Press Enter to continue...")