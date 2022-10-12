from json import load
from utils import *
from visualize_data import *
from computations import *
import numpy as np

def example_uses():
    #Load first dataset
    X_train, y_train = load_data("data/ex2data1.txt")

    #Visualize data
    viz(X_train,y_train,"Admitted","Not admitted","Exam 1 score","Exam 2 score")

    m, n = X_train.shape

    # Compute and display cost with w initialized to zeroes
    initial_w = np.zeros(n)
    initial_b = 0.
    cost = compute_cost(X_train, y_train, initial_w, initial_b)
    print('Cost at initial w (zeros): {:.3f}'.format(cost))

    # Compute and display gradient with w initialized to zeroes
    initial_w = np.zeros(n)
    initial_b = 0.

    dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
    print(f'dj_db at initial w (zeros):{dj_db}' )
    print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )

    np.random.seed(1)
    intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
    initial_b = -8


    # Some gradient descent settings
    iterations = 10000
    alpha = 0.001

    w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                    compute_cost, compute_gradient, alpha, iterations, 0)

    #From utils, visualize first decision boundary
    plot_decision_boundary(w, b, X_train, y_train)

    #Compute accuracy on our training set
    p = predict(X_train, w,b)
    print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

    #Load second dataset
    X_train, y_train = load_data("data/ex2data2.txt")

    #Visualize data
    viz(X_train,y_train[:],"Accepted","Rejected","Microchip Test 1","Microchip Test 2")

    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5
    cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

    print("Regularized cost :", cost)

    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1])-0.5
    initial_b = 1.

    # Set regularization parameter lambda_ to 1 (you can try varying this)
    lambda_ = 0.01;                                          
    # Some gradient descent settings
    iterations = 10000
    alpha = 0.01

    w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                        compute_cost_reg, compute_gradient_reg, 
                                        alpha, iterations, lambda_)

    print("Regularized cost :", cost)

    #From utils, visualize second decision boundary
    plot_decision_boundary(w, b, X_mapped, y_train)

    #Compute accuracy on the training set
    p = predict(X_mapped, w, b)

    print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

if __name__ == "__main__":
    example_uses()