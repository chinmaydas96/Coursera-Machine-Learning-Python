import numpy as np

from Submission import Submission
from Submission import sprintf

homework = 'regularized-linear-regression-and-bias-variance'

part_names = [
  'Regularized Linear Regression Cost Function',
  'Regularized Linear Regression Gradient',
  'Learning Curve',
  'Polynomial Feature Mapping',
  'Validation Curve'
  ]

srcs = [
  'linearRegCostFunction.py',
  'linearRegCostFunction.py',
  'learningCurve.py',
  'polyFeatures.py',
  'validationCurve.py'
  ]


def output(part_id):
    # Random Test Cases
    X = np.column_stack((np.ones(10),
                          (np.sin(np.arange(1, 16, 1.5))),
                          (np.cos(np.arange(1, 16, 1.5)))))
    y = np.sin(np.arange(1, 30, 3))

    Xval = np.column_stack((np.ones(10),
                          (np.sin(np.arange(0, 14, 1.5))),
                          (np.cos(np.arange(0, 14, 1.5)))))
    yval = np.sin(np.arange(1,11))

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        J, _ = func(X, y, np.array([0.1, 0.2, 0.3]), 0.5)
        return sprintf('%0.5f ', J)
    elif part_id == 2:
        _, grad = func(X, y, np.array([0.1, 0.2, 0.3]), 0.5)
        return sprintf('%0.5f ', grad)
    elif part_id == 3:
        error_train, error_val = func(X, y, Xval, yval, 1)
        return sprintf('%0.5f ', np.hstack((error_train, error_val)))
    elif part_id == 4:
        X_poly = func(X[1, :].T, 8)
        return sprintf('%0.5f ', X_poly)
    elif part_id == 5:
        lambda_vec, error_train, error_val = func(X, y, Xval, yval)
        return sprintf('%0.5f', np.hstack((lambda_vec, error_train, error_val)))

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print message
