# Author : Jai Sharma
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv('AAPL.csv')

# define data variables X, y, m, and alpha
y = df.iloc[1:50, 1].values.reshape(-1, 1).reshape(-1,1)
#    length of the dataset
m = len(y)
#    learning rate alpha
alpha = 1.95
#    our x values are the day number
X = np.zeros((m,1))  # we start with an array of zeros
for i in range(m): # and we fill in the elements as 1,2,3, etc.
    X[i][0]=i+1

# In order to perform gradient descent, we normalize the X and y values
meanX = np.mean(X)
rangeX = np.ptp(X)
meany = np.mean(y)
rangey = np.ptp(y)
Xnorm = (X-meanX)/rangeX
ynorm = (y-meany)/rangey


# initialize theta values at zero
# Remember, we need two theta values. One must be the slope, and the other must be the y intercept.
theta = np.zeros((2,1))

# it is hard to transpose a one dimensional array/vector, so we make our own function
# this will be useful later on
def Transpose(vector):
    # define the dimesions of the given vector
    rows = vector.shape[0]
    columns = vector.shape[1]
    return vector.reshape(columns,rows)

# We define another function to compute the cost function, given theta values,
# X values, and y values.
def CostFunction(Xvals,yvals,thetavals):
    # Sidnote: Some of the processes here will involve matricies and vectors,
    # but you need not worry if you do not know about vectors or matricies yet.
    # Just make sure you follow the logic behind what we are doing.

    # We first append a vector of ones for our bias unit (aka for our y intercept)
    Xvals = np.hstack((np.ones((Xvals.shape[0],1)), Xvals))
    # Now we compute the hypothesis using matrix multiplication.
    hx = Xvals.dot(thetavals)
    # Now that we have our hypothesis, we can compute the cost function.
    # Remember, the cost function is the mean squared error.
    return (1/(2*m))*sum((hx-yvals)**2)

# Now, we compute the gradient of the theta values.
def Gradient(Xvals,yvals,thetavals):
    # Once again, we append the ones to our Xvals and compute the hypothesis.
    Xvals = np.hstack((np.ones((Xvals.shape[0],1)), Xvals))
    hx = Xvals.dot(thetavals)
    # We now compute the gradient of the theta values.
    return thetavals - alpha * (1/m)* (Xvals.T) @ (hx-yvals)


# Now that we have implemented linear regression, let us move onto polynomial regression.
# Specifically, we try to fit a quintic polynomial to the data.

# First, we define our X again to include the quadratic, cubic, quartic, and quintic terms.
Xnorm = Xnorm[:,0]
Xnorm = np.c_[Xnorm,Xnorm**2,Xnorm**3,Xnorm**4,Xnorm**5]
# We must also define theta again. We use a 6 dimensional vector for the:
# 1) Constant Term; 2) x term; 3) x^2 term; 4) x^3 term; 5) x^4 term; 6) x^5 term.
theta = np.zeros((6,1))

# Now, we just run gradient descent like before. This time, we do 10000 iterations
# to fit polynomial regression nicely.
for i in range(10000):
    # First, we print the theta values.
    print("Theta: ")
    print(theta)
    # We print the cost function as a general indicator of how our program is running.
    print("Cost Function: ")
    print(CostFunction(Xnorm,ynorm,theta))
    # Adjust theta values according to the gradient.
    theta = Gradient(Xnorm,ynorm,theta)

# We now plot our results, like before.
# Compute the hypothesis using the normalized X values, appending a vector of ones to Xnorm for the bias unit.
hx = np.hstack((np.ones((Xnorm.shape[0],1)), Xnorm)).dot(theta)
# Now, we de-normalize the predictions to get our predictions.
hx = (hx*rangey)+meany
# Plot the original data points.
plt.plot(X,y)
# Plot the hypothesis.
plt.plot(X,hx)
# Show the plot.
plt.show()

# That is a simple introduction to regression using gradient descent and machine learning. I hope you had fun. The end!
