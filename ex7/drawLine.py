import matplotlib.pyplot as plt
import numpy as np

from show import show

def drawLine(p1, p2, varargin):
    """Draws a line from point p1 to point p2 and holds the
    current figure
    """

    plt.plot(np.column_stack(p1(1), p2(1)), np.column_stack(p1(2), p2(2)), varargin)
    show()