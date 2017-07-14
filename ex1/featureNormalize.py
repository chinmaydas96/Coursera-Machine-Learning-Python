import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    
    mu = np.mean(X[:, :1])
    sigma = np.std(X[:, :1])

    mu1 = np.mean(X[:, 1:])
    sigma1 = np.std(X[:, 1:])

    x_ = (X[:, :1] - mu) / sigma
    x1_ = (X[:, 1:] - mu1) / sigma1

    X_norm = np.append(x_, x1_, axis=1)
    return X_norm

# ============================================================

    return X_norm, mu, sigma
