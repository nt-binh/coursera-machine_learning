import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#--------DATA1--------#
"""def gradientDescent(dataX, dataY, theta, iter, alpha):
    J_history = []
    size = len(dataY)
    #start iteration on alpha
    for i in range(iter):
        theta = theta - (alpha / size) * (np.dot(dataX, theta) - dataY).dot(dataX)
        J_history.append(computeCost(dataX, dataY, theta))
    return theta, J_history

def computeCost(dataX, dataY, theta):
    size = len(dataX)
    J = 0
    #Hypothesis
    h = np.dot(dataX, theta)
    J = (1 / (2 * size)) * np.sum(np.square(h - dataY))
    return J


def plotData(dataX, dataY, theta):
    plt.style.use("fivethirtyeight")
    plt.plot(dataX[:, 1], dataY, "bo")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000")
    plt.plot(dataX[:, 1], np.dot(dataX, theta), "r-")
    plt.legend(["Training data", "Linear Regression"])
    plt.tight_layout()

if __name__ == "__main__":
    data = np.genfromtxt("ex1data1.txt", delimiter = ",")
    dataX = data[:, 0]
    dataY = data[:, 1]
    #length of vectors dataX, dataY
    size = len(dataY)
    #adding the ones vector to dataX
    dataX = np.stack((np.ones(size), dataX), axis = 1)
    theta = np.zeros(2)
    #compute the cost with theta1 = 0, theta2 = 0
    print(f"With theta1 = {theta[0]} and {theta[1]}, cost is {computeCost(dataX, dataY, theta)}")
    iter = 1500
    alpha = 0.01
    theta, J_history = gradientDescent(dataX, dataY, theta, iter, alpha)
    print(f"After gradient descent, theta1 = {theta[0]} and theta2 = {theta[1]}")
    plotData(dataX, dataY, theta)

    #Plotting examples
    ex1 = np.array([1, 3.5])
    ex2 = np.array([1, 7])
    plt.plot(ex1[1], np.dot(ex1, theta), "go")
    plt.plot(ex2[1], np.dot(ex2, theta), "yo")
    plt.show()
    
"""

#--------DATA2---------#
#LINEAR REGRESSION WITH MULTIPLE VARIABLES#
def featureNormalization(arr):
    return (arr - np.mean(arr, axis = 0)) / np.std(arr, axis = 0)

def gradientDescent(dataX, dataY, theta, iter, alpha):
    J_history = []
    size = len(dataY)
    for i in range(iter):
        theta = theta - (alpha / size) * (np.dot(dataX, theta) - dataY).dot(dataX)
        J_history.append(computeCost(dataX, dataY, theta))
    return theta, J_history

def computeCost(dataX, dataY, theta):
    #hypothesis
    h = np.dot(dataX, theta)
    size = len(dataY)
    #return the cost function
    return (1 / (2 * size)) * np.sum(np.square(h - dataY))

def plotLearningRate(J_history, alpha):
    plt.plot(alpha, J_history, "b")
    plt.tight_layout()
    plt.yscale("log")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")

if __name__ == "__main__":
    data = np.genfromtxt("ex1data2.txt", delimiter = ",")
    dataX = data[:, :2]
    dataY = data[:, 2]
    size = len(dataY)

    #feature normalization
    dataX = featureNormalization(dataX)
    dataY = featureNormalization(dataY)

    #combine vector 1's to dataX
    one = np.ones(size).reshape(size, 1)
    dataX = np.concatenate((one, dataX), axis = 1)
    theta = np.arange(3)
    
    #initialize theta
    theta = np.zeros(3)

    #gradient descent
    theta, J_history = gradientDescent(dataX, dataY, theta, 1000, 0.5)
    alpha = np.linspace(0,2,11)
    print(alpha)
    
    
