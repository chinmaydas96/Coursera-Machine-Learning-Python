import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('ex1data1.txt', delimiter=',')

def plotData(data):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    # print data
    row1 = data[:,:1]
    row2 = data[:,1:]
    plt.scatter(row1,row2,c='b')
    # plt.show()	# open a new figure window

# ============================================================
# plotData(data)