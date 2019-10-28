#
# Optional exercise of exercise 1
#


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Import training data2
data = pd.read_csv("ex1data2.txt", names = ["size", "bedrooms", "price"])
x_raw = data[["size", "bedrooms"]].values
y_raw = data["price"].values

# Nomalize features
def featureNomalize(x_raw):
    mean = np.mean(x_raw, axis = 0)
    std = np.std(x_raw, axis = 0)
    x = (x_raw - mean)/std

    return x

# Set parameters
m = y_raw.shape[0]
x = np.column_stack((np.ones(m), featureNomalize(x_raw)))
y = y_raw.reshape(m,1)
theta = np.zeros((np.size(x, axis=1),1))
iteration = 1500 # Iteration times
alpha = 0.01 # Learning rate

# Define multivariate cost function
def computeCostMulti(x, y, theta):
    hypothesis = np.dot(x, theta)
    cost = np.dot((hypothesis - y).T, hypothesis - y)/(2*m)

    return cost

# Define multivariate gradient descent function
def gradientDescentMulti(x, y, theta, alpha, iteration):
    cost_list = np.zeros(iteration)

    for i in range(iteration):
        cost_list[i] = computeCostMulti(x, y, theta)
        hypothesis = np.dot(x, theta)
        partial_derivative = np.dot(x.T, hypothesis - y)*alpha/m
        theta = theta - partial_derivative

    return theta, cost_list

# Plot costs trend 
def plotCostsTrend(x, y, theta, iteration):
    iteration_list = np.arange(iteration)
    theta_opt, cost_list = gradientDescentMulti(x, y, theta, alpha, iteration)
    
    figure2 = plt.figure(2)
    plt.plot(iteration_list, cost_list, color="b")
    plt.title("Costs trend")
    plt.xlabel("Iteration tims")
    plt.ylabel("Cost")
    plt.show()

plotCostsTrend(x, y, theta, iteration)

# Normal equation
def normalEqn(x, y):
    theta_opt = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    
    return theta_opt

# Prediction for 1650 square foot with 3 bedrooms
x_test = np.array([1, 1650, 3])

print("Use gradient descent: ")
theta_opt, cost_list = gradientDescentMulti(x, y, theta, alpha, iteration)
print("The price should be", np.dot(x_test, theta_opt))

print("Use normal equation: ")
theta_opt = normalEqn(x, y)
print("The price should be", np.dot(x_test, theta_opt))

