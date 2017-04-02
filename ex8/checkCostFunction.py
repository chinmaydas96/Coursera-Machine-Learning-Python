import numpy as np
from ex4.computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc

def checkCostFunction(Lambda=0):
    """Creates a collaborative filering problem
    to check your cost function and gradients, it will output the
    analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient
    computations should result in very similar values.
    """

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.where(np.random.random_sample(Y.shape) > 0.5, True, False)] = 0
    R = np.zeros(Y.shape)
    R[np.where(Y != 0, True, False)] = 1

    ## Run Gradient Checking
    X = np.random.random_sample(X_t.shape)
    Theta = np.random.random_sample(Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

   # Unroll parameters
    params = np.hstack((X.T.flatten(), Theta.T.flatten()))

    costFunc = lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, Lambda)

    def costFunc_w(t):
        Jgrad = costFunc(t)
        return Jgrad

    numgrad = computeNumericalGradient(costFunc_w, params)

    cost, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda)


    print np.column_stack((numgrad, grad))

    print 'The above two columns you get should be very similar.\n' \
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n'

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print 'If your backpropagation implementation is correct, then\n ' \
          'the relative difference will be small (less than 1e-9). \n' \
          '\nRelative Difference: %g\n' % diff

