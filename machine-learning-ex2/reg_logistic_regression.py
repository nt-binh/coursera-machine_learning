import numpy as np 
import matplotlib.pyplot as plt 
import logistic_regression as lg
from scipy import optimize 

def plotData(dataX, dataY, degree = 6):
    pos = (dataY == 1)
    neg = (dataY == 0)

    plt.plot(dataX[pos, 0], data[pos, 1], "bo")
    plt.plot(dataX[neg, 0], data[neg, 1], "r*", ms = 10)
    legends = ["y = 1", "y = 0"]

    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend(legends)
    plt.tight_layout()
    plt.show()

def featureMapping(dataX):
    """
    One way to fit the data better is to create more features frome ach data point
    In the function featureMapping, we will map the features into all polynomial
    terms of x1 and x2 up to the 6th power
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    
    dataX1 = dataX[:, 1]
    dataX2 = dataX[:, 2]
    size = dataX.shape[0]

    for i in range(2,7):
        for j in range(i + 1):
            polynomial = dataX1 ** (i-j) * dataX2 ** (i)
            polynomial = polynomial.reshape(size, 1)
            dataX = np.hstack((dataX, polynomial))
    
    return dataX


def costFunction(theta, dataX, dataY, lambdaNo):
    h = lg.sigmoid(dataX, theta)
    size = len(dataY)
    
    temp = theta
    temp[0] = 0
    J = (1 / size) * np.sum(-np.dot(dataY, np.log(h)) - np.dot(1 - dataY, np.log(1 - h))) + (lambdaNo / (2*size)) * np.sum(theta**2)
    grad = (1 / size) * (h - dataY).dot(dataX)
    grad = grad + (lambdaNo / size) * temp
    return J, grad
    

def predict(dataX, theta):
    size = dataX.shape[0]
    p = np.zeros(size)
    p = np.round(lg.sigmoid(dataX, theta))
    return p

if __name__ == "__main__":
    data = np.genfromtxt("ex2data2.txt", delimiter = ",")
    dataX = data[:, :2]
    dataY = data[:, 2]
    size = dataX.shape[0]

    ones = np.ones((size, 1))
    dataX = np.concatenate((ones, dataX), axis = 1)
    plotData(dataX[:,1:], dataY)
    dataX = featureMapping(dataX)

    theta = np.zeros(dataX.shape[1])
    lambdaNo = 0.1
    #try to optimize the theta using 
    options = {"maxiter" : 400}
    res = optimize.minimize(costFunction, 
                            theta,
                            (dataX, dataY, lambdaNo),
                            jac = True,
                            method = "TNC",
                            options = options)

    J = res.fun 
    grad = res.x

    p = predict(dataX, grad)
    print(np.mean(p == dataY) * 100)
    

    
