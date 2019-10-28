#
# Exercise 2 of ML Andrew NG
#

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
import tensorflow as tf


# Import training data1
data = pd.read_csv("ex2data1.txt", names=["exam1", "exam2", "decision"])
x_raw = data[["exam1", "exam2"]].values
y_raw = data["decision"].values

# Plot training data1
def plotData(x, y):
    figure1 = plt.figure(1)
    plt.scatter(x[y[:] == 1, 0], x[y[:] == 1, 1], color="k", marker="+", label="admitted")
    plt.scatter(x[y[:] == 0, 0], x[y[:] == 0, 1], color="y", marker="o", label="denied")
    plt.title("Training data1")
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    plt.legend()
    plt.show()

plotData(x_raw, y_raw)

# Set parameters
m = y_raw.shape[0]
x = np.column_stack((np.ones(m), x_raw))
y = y_raw.reshape(m, 1)
n = x.shape[1]
theta_init = np.zeros(n) # Initialized as 1-D array in order to fit the opt.minmize() function

# Define Sigmoid function
def sigmoid(z):
    g = 1/(1 + np.exp(-z))

    return g

# Define cost function
def costFunction(theta, x, y): # 'theta' must be put in the first position as a 1-D array!
    theta = theta.reshape(n, 1) # Reshape to (n,1) matrix
    h = sigmoid(np.dot(x, theta))
    cost = (np.dot((-y).T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))/m

    return cost

# print(costFunction(x, y, theta)) # Around 0.693 is correct

# Define gradient function 
def gradient(theta, x, y): # 'theta' must be put in the first position as 1-D array!
    theta = theta.reshape(n, 1) # Reshape to (n,1) matrix
    h = sigmoid(np.dot(x, theta))
    partial_derivative = np.dot((h - y).T, x)/m

    return partial_derivative.flatten() # Return must be 1-D array as well!

# Use opt.minimize() to substitute fminunc() from Octave
result = opt.minimize(fun=costFunction, x0=theta_init, args=(x, y), method="Newton-CG", jac=gradient) # Optimal cost is around 0.203
theta_opt = result['x'].reshape(n, 1)

# Plot the decision boundary
def plotDecisionBoundary(x, y, theta_opt):
    exam1 = np.arange(30, 100, 3)
    exam2 = (-theta_opt[2]*exam1 - theta_opt[0])/theta_opt[1]
    
    figure2 = plt.figure(2)
    plt.scatter(x[y[:] == 1, 0], x[y[:] == 1, 1], color="k", marker="+", label="admitted")
    plt.scatter(x[y[:] == 0, 0], x[y[:] == 0, 1], color="y", marker="o", label="denied")
    plt.plot(exam1, exam2, color="r", label="Decision Boundary")
    
    plt.title("Training data1 with decision boundary")
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    plt.legend()
    plt.show()

plotDecisionBoundary(x_raw, y_raw, theta_opt)

# Prediction of test data
print("For a student with an Exam 1 score of 45 and an Exam 2 score of 85: ")
scores = np.array([1, 45, 85])
scores = scores.reshape(1, 3)
print("The possibility of admission is ", sigmoid(np.dot(scores, theta_opt))[0][0]) # The posibility should be around 0.776

