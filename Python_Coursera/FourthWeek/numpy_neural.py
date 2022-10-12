import numpy as np
from utils import *

def my_dense(a_in, W, b, g):
    """
    Computes dense layer with numpy
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w,a_in) + b[j]
        a_out[j] = g(z)
    return(a_out)

#Builds a 3-layer network using above my_dense

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)

#Vectorized version of my_dense
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
    Z = np.matmul(A_in,W) + b
    A_out = g(Z)
    return(A_out)

#Builds a 3-layer network using vectorized my_dense
def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return(A3)

#Visualize misclassifications
def misclassified(Prediction,X,y):
    Yhat = (Prediction >= 0.5).astype(int)
    print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])

    #Visualize a mis-classified example
    fig = plt.figure(figsize=(1, 1))
    errors = np.where(y != Yhat)
    random_index = errors[0][0]
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    plt.imshow(X_random_reshaped, cmap='gray')
    plt.title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
    plt.axis('off')
    plt.show()