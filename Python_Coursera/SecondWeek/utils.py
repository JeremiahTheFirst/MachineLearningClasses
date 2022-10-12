import numpy as np
import os

def load_data():
    folder = os.path.abspath('SecondWeek/data/')
    filename = os.path.abspath(os.path.join(folder,'ex1data1.txt'))
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y