import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

def gradientDescent(dataX, dataY, theta, iter, alpha):
    J_history = []
    size = len(dataY)
    #start iterating using learning rate "alpha"
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
    #choose the "fivethirtyeight" style
    plt.style.use("fivethirtyeight")
    
    #plot the training data
    plt.plot(dataX[:, 1], dataY, "bo")
    
    #label two axises
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000")
    
    #plot the test data
    plt.plot(dataX[:, 1], np.dot(dataX, theta), "r-")
    plt.legend(["Training data", "Linear Regression"])
    plt.tight_layout()

if __name__ == "__main__":
    #fetching data from .txt file
    data = np.genfromtxt("ex1data1.txt", delimiter = ",")
    
    #split into two categories
    dataX = data[:, 0]
    dataY = data[:, 1]
    size = len(dataY)   #length of vectors dataX, dataY
    
    #adding the ones vector to dataX
    dataX = np.stack((np.ones(size), dataX), axis = 1)
    theta = np.zeros(2)
    
    #compute the cost with theta1 = 0, theta2 = 0
    print(f"With theta1 = {theta[0]} and {theta[1]}, cost is {computeCost(dataX, dataY, theta)}")
    iter = 1500
    alpha = 0.01
    
    #go through gradient descent
    theta, J_history = gradientDescent(dataX, dataY, theta, iter, alpha)
    print(f"After gradient descent, theta1 = {theta[0]} and theta2 = {theta[1]}")
    plotData(dataX, dataY, theta)

    #plotting examples (test data)
    ex1 = np.array([1, 3.5])
    ex2 = np.array([1, 7])
    plt.plot(ex1[1], np.dot(ex1, theta), "go")
    plt.plot(ex2[1], np.dot(ex2, theta), "yo")
    plt.show()

