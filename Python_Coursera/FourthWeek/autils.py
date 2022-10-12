import numpy as np
import os

folder = os.path.abspath('FourthWeek/data/')

def load_data():
    filename = os.path.abspath(os.path.join(folder,'X.npy'))
    X = np.load(filename)
    filename = os.path.abspath(os.path.join(folder,'y.npy'))
    y = np.load(filename)
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    filename = os.path.abspath(os.path.join(folder,'w1.npy'))
    w1 = np.load(filename)
    filename = os.path.abspath(os.path.join(folder,'b1.npy'))
    b1 = np.load(filename)
    filename = os.path.abspath(os.path.join(folder,'w2.npy'))
    w2 = np.load(filename)
    filename = os.path.abspath(os.path.join(folder,'b2.npy'))
    b2 = np.load(filename)
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
