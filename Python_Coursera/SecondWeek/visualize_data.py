import matplotlib.pyplot as plt

def visualize_data(title,ylabel,xlabel):

    plt.scatter(x_train, y_train, marker='x', c='r') 

    # Set the title
    plt.title(title)
    # Set the y-axis label
    plt.ylabel(ylabel)
    # Set the x-axis label
    plt.xlabel(xlabel)
    plt.show()

