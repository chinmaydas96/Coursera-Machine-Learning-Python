import numpy as np

from Submission import Submission
from Submission import sprintf
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from predict import predict
from ex2.gradientFunctionReg import gradientFunctionReg

homework = 'multi-class-classification-and-neural-networks'

part_names = [
  'Regularized Logistic Regression',
  'One-vs-All Classifier Training',
  'One-vs-All Classifier Prediction',
  'Neural Network Prediction Function',
  ]

srcs = [
  'lrCostFunction.py',
  'oneVsAll.py',
  'predictOneVsAll.py',
  'predict.py',
  ]


def output(part_id):
    # Random Test Cases
    X = np.column_stack((np.ones(20),
                          (np.exp(1) * np.sin(np.linspace(1, 20, 20))),
                          (np.exp(0.5) * np.cos(np.linspace(1, 20, 20)))))
    y = np.sin(X[:,0] + X[:,1]) > 0

    Xm = np.array([[-1,-1],[-1,-2],[-2,-1],[-2,-2],[1,1],[1,2],[2,1],[2,2],[-1,1],
          [-1,2],[-2,1],[-2,2],[1,-1],[1,-2],[-2,-1],[-2,-2]]).reshape((16,2))
    ym = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]).reshape(16,1)
    t1 = np.sin(np.array(range(1,24,2)).reshape(3,4).T)
    t2 = np.cos(np.array(range(1,40,2)).reshape(5,4).T)

    fname = srcs[part_id-1].rsplit('.',1)[0]
    mod = __import__(fname, fromlist=[fname], level=1)
    func = getattr(mod, fname)

    if part_id == 1:
        J = lrCostFunction(np.array([0.25, 0.5, -0.5]), X, y, 0.1)
        grad = gradientFunctionReg(np.array([0.25, 0.5, -0.5]), X, y, 0.1)
        return sprintf('%0.5f ', np.hstack((J, grad)).tolist())
    elif part_id == 2:
        return sprintf('%0.5f ', oneVsAll(Xm, ym, 4, 0.1))
    elif part_id == 3:
        return sprintf('%0.5f ', predictOneVsAll(t1, Xm))
    elif part_id == 4:
        return sprintf('%0.5f ', predict(t1, t2, Xm))

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print message

