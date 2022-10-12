import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import warnings

# load dataset
X, y = load_data()

def seq_mod():
    model = Sequential(
        [               
            tf.keras.Input(shape=(400,)),    #specify input size (not required)
            Dense(25, activation='sigmoid'), 
            Dense(15, activation='sigmoid'), 
            Dense(1,  activation='sigmoid')
        ], name = "my_model" 
    )    
    model.summary()

    #Looking at the parameters from the model summary and where those numbers come from
    L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
    L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
    L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )
    [layer1, layer2, layer3] = model.layers
    #### Examine Weights shapes
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    #Define loss and fit to training data
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X,y,
        epochs=20
    )

    #Example predictions
    prediction = model.predict(X[0].reshape(1,400))  # a zero
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1,400))  # a one
    print(f" predicting a one:  {prediction}")

    #Compare predictions to labels (accuracy)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()
    return X,y,layer1,layer2,layer3

if __name__ == "__main__":
    seq_mod()