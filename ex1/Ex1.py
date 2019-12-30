#
# Exercise 1 of ML Andrew NG 
#

import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Import training data1
data = pd.read_csv("ex1data1.txt", names = ["population", "profit"])
x_raw = data["population"].values
y_raw = data["profit"].values

# Plot ex1data1
figure1 = plt.figure(1)
plt.scatter(x_raw, y_raw, color = "b", marker = "x")
plt.title("trainning data1")
plt.xlabel("poplution")
plt.ylabel("profit")
plt.show()

# Set parameters 
m = y_raw.shape[0]
x = np.column_stack((np.ones(m), x_raw)) # Add x_0 = 1
y = y_raw.reshape(m, 1)
theta = np.zeros((2, 1)) # Theta initialized as [[0], [0]]
iteration = 1500 # Gradient descent iteration times
alpha = 0.01 # Gradient descent coefficient

# Define cost function
def computeCost(x, y, theta):
    hypothesis = np.dot(x, theta) # Hypothesis "h_theta of x"
    error = hypothesis - y
    error_square_sum = np.sum(error*error)
    cost = error_square_sum/(2*m)

    return cost

#print(computeCost(x, y, theta)) # The outcome of cost is 32.072733877455676 using theta = np.zeros((2, 1))

# Define gradient descent function
def gradientDescent(x, y, theta, iteration):
    for i in range(iteration):
        #print(theta)
        hypothesis = np.dot(x, theta)
        partial_derivative = alpha/m*np.dot(x.T, (hypothesis - y))
        #print(partial_derivative)
        theta = theta - partial_derivative

    return theta

# Plot each cost in iterations
def plotCosts(x, y, theta, iteration):
    iteration_matrix = np.arange(iteration)
    cost_matrix = np.zeros(iteration)
    
    for i in range(iteration):
        cost_matrix[i] = computeCost(x, y, theta)
        
        hypothesis = np.dot(x, theta)
        partial_derivative = alpha/m*np.dot(x.T, (hypothesis - y))
        theta = theta - partial_derivative
    
    figure2 = plt.figure(2)
    plt.plot(iteration_matrix, cost_matrix, color="g")
    plt.title("costs trend")
    plt.xlabel("iteration times")
    plt.ylabel("cost")
    plt.show()

plotCosts(x, y, theta, iteration) # Execute plotCosts function
 
# Plot the optimal hypothesis and give the linear regression formula
theta_opt = gradientDescent(x, y, theta, iteration)
hypothesis_opt = np.dot(x, theta_opt)

figure3 = plt.figure(3)
plt.plot(x_raw, hypothesis_opt, color="r", label='linear regression')
plt.scatter(x_raw, y_raw, color="b", marker="x", label='training data1')
plt.title("linear regression")
plt.xlabel("population")
plt.ylabel("profit")
plt.legend()
plt.show()

print("Linear regression of ex1data1.txt is: ")
print("h = ", theta_opt[0], "+", theta_opt[1], "* x1\n")

# Prediction on 35,000 and 70,000 people
print("With 35,000 people, we predict the profit should be: ")
p1 = np.array([1, 3.5])
print(10000*np.dot(p1, theta_opt), "\n")

print("With 70,000 people, we predict the profit should be: ")
p2 = np.array([1, 7])
print(10000*np.dot(p2, theta_opt), "\n")

