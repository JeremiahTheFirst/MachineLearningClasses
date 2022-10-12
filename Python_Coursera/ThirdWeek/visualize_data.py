import matplotlib.pyplot as plt
from utils import *

def viz(xvals,yvals,poslab,neglab,xlab,ylab):

    X_train = xvals
    y_train = yvals
    # Plot examples
    plot_data(X_train, y_train[:], pos_label=poslab, neg_label=neglab)

    # Set the y-axis label
    plt.ylabel(ylab) 
    # Set the x-axis label
    plt.xlabel(xlab) 
    plt.legend(loc="upper right")
    plt.show()