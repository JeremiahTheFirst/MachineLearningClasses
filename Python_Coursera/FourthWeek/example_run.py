import numpy_neural
from sequential_model import seq_mod
from numpy_neural import *

def doit():
    X,y,layer1,layer2,layer3 = seq_mod()
    W1_tmp,b1_tmp = layer1.get_weights()
    W2_tmp,b2_tmp = layer2.get_weights()
    W3_tmp,b3_tmp = layer3.get_weights()
    Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    misclassified(Prediction,X,y)

if __name__ == "__main__":
    doit()