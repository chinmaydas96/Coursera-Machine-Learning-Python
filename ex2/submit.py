import numpy as np

from Submission import Submission
from Submission import sprintf

__all__ = ['submit']

homework = 'logistic-regression'

part_names = [
    'Sigmoid Function',
    'Logistic Regression Cost',
    'Logistic Regression Gradient',
    'Predict',
    'Regularized Logistic Regression Cost',
    'Regularized Logistic Regression Gradient',
    ]

srcs = [
    'sigmoid.py',
    'costFunction.py',
    'gradientFunction.py',
    'predict.py',
    'costFunctionReg.py',
    'gradientFunctionReg.py',
    ]


def output(part_id):
    X = np.column_stack((np.ones(20),
                          (np.exp(1) * np.sin(np.linspace(1, 20, 20))),
                          (np.exp(0.5) * np.cos(np.linspace(1, 20, 20)))))
    Y = np.sin(X[:,0] + X[:,1]) > 0

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        return sprintf('%0.5f ', func(X))
    elif part_id == 2:
        return sprintf('%0.5f ', func(np.array([0.25, 0.5, -0.5]), X, Y))
    elif part_id == 3:
        return sprintf('%0.5f ', func(np.array([0.25, 0.5, -0.5]), X, Y))
    elif part_id == 4:
        return sprintf('%0.5f ', func(np.array([0.25, 0.5, -0.5]), X))
    elif part_id == 5:
        return sprintf('%0.5f ', func(np.array([0.25, 0.5, -0.5]), X, Y, 0.1))
    elif part_id == 6:
        return sprintf('%0.5f ', func(np.array([0.25, 0.5, -0.5]), X, Y, 0.1))

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print message
