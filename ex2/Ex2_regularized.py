#
# Regularized logistic regression
#

import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
import tensorflow as tf


# Import training data2
data = pd.read_csv("ex2data2.txt", names=["test1", "test2", "decision"])
x_raw = data[["test1", "test2"]].values
y_raw = data["decision"].values

# Plot training data2 
def plotData(x, y):
    figure1 = plt.figure(1)
    plt.scatter(x[y[:]==0, 0], x[y[:]==0, 1], color="y", marker="o", label="rejected")
    plt.scatter(x[y[:]==1, 0], x[y[:]==1, 1], color="k", marker="+", label="accepted")
    plt.title("Training data2")
    plt.xlabel("test1")
    plt.ylabel("test2")
    plt.legend()
    plt.show()

plotData(x_raw, y_raw)

# Feature mapping
def mapFeature(x):
    m,n = np.shape(x)
    x_mapped = np.ones((m, 28))

    for i in range(m):
        col = 0
        for j in range(7):
            for d in range(j):
                x_mapped[i][col+d] = np.power(x[i][0], d)*np.power(x[i][1], j-d)
            col = col + j + 1

    return x_mapped

# Set parameters
x = mapFeature(x_raw)
m,n = np.shape(x)
y = y_raw.reshape(m, 1)
theta_init = np.zeros(n)
l = 1 # lambda rate

# Define sigmoid function
def sigmoid(z):
    g = 1/(1 + np.exp(-z))

    return g

# Define cost function
def costFunctionReg(theta, x, y, l):
    theta = theta.reshape(n, 1)
    h = sigmoid(np.dot(x, theta))
    cost = (np.dot((-y).T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))/m + l/(2*m)*np.dot(theta.T, theta)

    return cost

# Define gradient 
def gradient(theta, x, y, l):
    theta = theta.reshape(n, 1)
    h = sigmoid(np.dot(x, theta))
    partial_derivative = np.dot(x.T, h-y)/m + l/m*theta

    return partial_derivative.flatten()

#print(costFunctionReg(theta_init, x, y, l)) # The cost with theta_init is 0.69314718

# Use opt.minimize() to substitute fminunc() from Octave
result = opt.minimize(fun=costFunctionReg, x0=theta_init, args=(x, y, l), method="Newton-CG", jac=gradient)
print(result)

theta_opt = result['x']

# Plot decision boundary, 
# Instead of solving polynomial equation
# Just create a coridate x,y grid that is dense enough 
# Then find all those x times theta that is close enough to 0 and  plot them
def plotDecisionBoundary(theta, x, y):
    test1 = np.linspace(-1, 1.5, 1000)
    test2 = np.linspace(-1, 1.5, 1000)
    x1 = []
    x2 = []
    
    for p1 in test1:
        for p2 in test2:
            x1.append(p1)
            x2.append(p2)
    
    x_points = np.array([x1, x2]).T
    x_points_mapped = mapFeature(x_points)
    threshold = 0.01
    x_boundary = x_points[(np.abs(np.dot(x_points_mapped, theta)) < threshold).flatten()]
    
    figure2 = plt.figure(2)
    plt.scatter(x[y[:]==0, 0], x[y[:]==0, 1], color="y", marker="o", label="rejected")
    plt.scatter(x[y[:]==1, 0], x[y[:]==1, 1], color="k", marker="+", label="accepted")
    plt.plot(x_boundary[:, 0], x_boundary[:, 1], color='r', label="decision boundary")
    plt.title("Training data2 with decision boundary")
    plt.xlabel("test1")
    plt.ylabel("test2")
    plt.legend()
    plt.show()

plotDecisionBoundary(theta_opt, x_raw, y_raw)
