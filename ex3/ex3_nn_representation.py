#
# Exercise 3 of ML Andrew NG, Neural Network, Prediction
#

import numpy as np
import scipy.io
from matplotlib.pyplot import axis


# Import data from ex3data1.mat
data = scipy.io.loadmat("ex3data1.mat")
X, y = data["X"], data["y"]
m, n = X.shape
X = np.append(np.ones((m, 1)), X, axis=1)

weights = scipy.io.loadmat("ex3weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]

# Define sigmoid function
def sigmoid(X):
    return 1/(1 + np.exp(-X))
 
# Define prediction function
def predict_nn(theta1, theta2, X):
    hidden_layer = sigmoid(X.dot(theta1.T))
    m, n = hidden_layer.shape
    hidden_layer = np.append(np.ones((m, 1)), hidden_layer, axis=1)
     
    output_layer = sigmoid(hidden_layer.dot(theta2.T))
    
    predict = output_layer.argmax(axis=1) + 1
    
    return predict

# Predict the result using trained nn
print("The accuracy of neural network is: \n")

prediction = predict_nn(theta1, theta2, X)

print(np.mean((prediction == y.flatten()).astype(int))*100)