import numpy as np

from Submission import Submission
from Submission import sprintf

homework = 'anomaly-detection-and-recommender-systems'

part_names = [
  'Estimate Gaussian Parameters',
  'Select Threshold',
  'Collaborative Filtering Cost',
  'Collaborative Filtering Gradient',
  'Regularized Cost',
  'Regularized Gradient',
  ]

srcs = [
  'estimateGaussian.py',
  'selectThreshold.py',
  'cofiCostFunc.py',
  'cofiCostFunc.py',
  'cofiCostFunc.py',
  'cofiCostFunc.py',
  ]


def output(part_id):
    # Random Test Cases
    n_u = 3
    n_m = 4
    n = 5
    X = np.sin(np.arange(1,n_m*n+1)).reshape((n, n_m)).T
    Theta = np.cos(np.arange(1,n_u*n+1)).reshape((n, n_u)).T
    Y = np.sin(np.arange(1,2.0*n_m*n_u,2.0)).reshape((n_u, n_m)).T
    R = Y > 0.5
    pval = np.hstack((np.abs(Y.T.flatten()), 0.001, 1.0))
    yval = np.hstack((R.T.flatten(), 1.0, 0.0)).astype('bool')
    params = np.hstack((X.T.flatten(), Theta.T.flatten()))

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        mu, sigma2 = func(X)
        return sprintf('%0.5f ', np.hstack((mu.T.flatten(), sigma2.T.flatten())))
    elif part_id == 2:
        bestEpsilon, bestF1 = func(yval, pval)
        return sprintf('%0.5f ', np.hstack((bestEpsilon, bestF1)))
    elif part_id == 3:
        J, grad = func(params, Y, R, n_u, n_m, n, 0.0)
        return sprintf('%0.5f ', J)
    elif part_id == 4:
        J, grad = func(params, Y, R, n_u, n_m, n, 0.0)
        return sprintf('%0.5f ', grad.T.flatten())
    elif part_id == 5:
        J, grad = func(params, Y, R, n_u, n_m, n, 1.5)
        return sprintf('%0.5f ', J)
    elif part_id == 6:
        J, grad = func(params, Y, R, n_u, n_m, n, 1.5)
        return sprintf('%0.5f ', grad.T.flatten())

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print message
