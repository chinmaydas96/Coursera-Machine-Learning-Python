import numpy as np

from Submission import Submission
from Submission import sprintf

homework = 'k-means-clustering-and-pca'

part_names = [
  'Find Closest Centroids (k-Means)',
  'Compute Centroid Means (k-Means)',
  'PCA',
  'Project Data (PCA)',
  'Recover Data (PCA)',
  ]

srcs = [
  'findClosestCentroids.py',
  'computeCentroids.py',
  'pca.py',
  'projectData.py',
  'recoverData'
  ]


def output(part_id):
    # Random Test Cases
    X = np.sin(np.arange(1,166)).reshape((11, 15)).T
    Z = np.cos(np.arange(1,122)).reshape((11, 11)).T
    C = Z[:5, :]
    idx = (np.mod(np.arange(1,16), 3)).T

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        idx = func(X, C)
        return sprintf('%0.5f ', idx[1]+1)
    elif part_id == 2:
        centroids = func(X, idx, 3)
        return sprintf('%0.5f ', centroids)
    elif part_id == 3:
        U, S, V = func(X)
        return sprintf('%0.5f ', abs(np.hstack((U.T.flatten(), S.T.flatten()))))
    elif part_id == 4:
        X_proj = func(X, Z, 5)
        return sprintf('%0.5f ', X_proj.T.flatten())
    elif part_id == 5:
        X_rec = func(X[:, :5], Z, 5)
        return sprintf('%0.5f ', X_rec.T.flatten())

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print message
