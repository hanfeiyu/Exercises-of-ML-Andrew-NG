#
# Exercise 3 of ML Andrew NG, multi-classification for logistic regression 
#

import numpy as np
import scipy.optimize as opt
import scipy.io
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import axis


# Import data from ex3data1.mat
data = scipy.io.loadmat("ex3data1.mat")
X, y = data["X"], data["y"]
m, n = X.shape
X = np.append(np.ones((m,1)), X, axis=1)

#print(X.shape)
  
# Create one image object from a single np array within 1x400 shape 
def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
        
    return square.T
    
# Display data 
def displayData(X):
    width, height = 20, 20
    nrows, ncols = 10, 10
    indices_to_display = random.sample(range(0, X.shape[0]), nrows*ncols)
    big_picture = np.zeros((height*nrows, width*ncols))
    irow, icol = 0, 0
      
    for idx in indices_to_display:
        if icol == ncols:
            irow = irow + 1
            icol = 0
         
        iimg = getDatumImg(X[idx])
        big_picture[irow * height : irow * height + iimg.shape[0], 
                    icol * width : icol * width + iimg.shape[1]] = iimg
        icol = icol + 1
        
    plt.imshow(big_picture, cmap=cm.Greys_r)
    plt.axis("off")
    plt.show()
    
#displayData(X)

# Define Sigmoid function
def sigmoid(X):
    return 1/(1 + np.exp(-X))

# Define regression cost function
def cost_function_reg(theta, X, y, lambda_reg):
    m, n = np.shape(X)
    theta = theta.reshape(n, 1)
    h = sigmoid(X.dot(theta))
    initial_theta = theta[0]
    theta[0] = 0
    
    j = (1/m)*(-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1 - h))) + lambda_reg/(2*m)*theta.T.dot(theta)                
    grad = (1/m)*X.T.dot(h-y) + lambda_reg/m*theta   
    theta[0] = initial_theta
    
    return j.flatten(), grad.flatten()

# Define function for one_vs_all method
def one_vs_all(X, y, num_labels, lambda_reg):
    m, n = X.shape
    all_theta = np.zeros((n, num_labels))
    
    for i in range(num_labels):
        initial_theta = np.zeros(n)
        find_theta = opt.minimize(fun = cost_function_reg, 
                                  method = "CG", 
                                  jac = True,
                                  x0 = initial_theta,
                                  args = (X, y == i+1, lambda_reg),
                                  options = {"maxiter": 50, "disp": False}
                                  ).x
        all_theta[:, i] = find_theta
        
    return all_theta                          
          
# Predict the result using one_vs_all function  
def predict_one_vs_all(theta, X):
    ps = sigmoid(X.dot(theta))
    p = ps.argmax(axis = 1) + 1
    #print(p)
    
    return p
    
# Test accuracy of the model using training dataset
num_labels = 10
lambda_reg = 0.1
all_theta = one_vs_all(X, y, num_labels, lambda_reg)

print("The accuracy of logistic regression is: \n")

prediction = predict_one_vs_all(all_theta, X)
print(prediction)

print(np.mean((prediction == y.flatten()).astype(int))*100)

            
            
            
    
