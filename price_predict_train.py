import tensorflow as tf
import numpy as np

def house_model():
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    xs = np.array([1,2,3,4,5,6],dtype='float')
    ys = np.array([100,150,200,250,300,350],dtype='float')
    
    # Define model
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1,])])
    
    # Compile model
    # Set the optimizer to Stochastic Gradient Descent and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mse')
    
    # Train model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    return model

"""Now a function that returns a compiled and trained model when invoked, use it to get the model to predict the price of houses: """

# Get trained model
model = house_model()

"""Now that model has finished training it is time to test it out! By running the next cell."""

new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)

