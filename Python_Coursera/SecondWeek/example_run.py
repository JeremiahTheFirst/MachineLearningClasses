from utils import load_data
from computations import *
import numpy as np

def example_uses():
    x_train, y_train = load_data()

    initial_w = 2
    initial_b = 1
    cost = compute_cost(x_train, y_train, initial_w, initial_b)
    print(f'Cost at weight 2: {cost:.3f}')

    initial_w = 0
    initial_b = 0
    tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w,initial_b)
    print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

    # some gradient descent settings
    iterations = 1500
    alpha = 0.01

    w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x_train[i] + b
    return predicted

if __name__ == "__main__":
    example_uses()