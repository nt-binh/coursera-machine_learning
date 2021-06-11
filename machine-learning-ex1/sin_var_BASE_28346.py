import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

def gradientDescent(dataX, dataY, theta, iter, alpha):
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
    plt.xlabel("Volume of acetone (mL)")
    plt.ylabel("Dielectric constant")
    plt.plot(dataX[:, 1], np.dot(dataX, theta), "r-")
    #plt.legend(["Training data", "Linear Regression"])
    plt.tight_layout()

if __name__ == "__main__":
    #data = np.genfromtxt("ex1data1.txt", delimiter = ",")
    #dataX = data[:, 0]
    #dataY = data[:, 1]
    dataX = np.arange(6)
    dataY = np.array([1.7, 1.838, 1.905, 2.043, 2.111, 2.224])
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
    plt.show()
"""
    #Plotting examples
    ex1 = np.array([1, 3.5])
    ex2 = np.array([1, 7])
    plt.plot(ex1[1], np.dot(ex1, theta), "go")
    plt.plot(ex2[1], np.dot(ex2, theta), "yo")
    plt.show()
data = np.genfromtxt("ex1data2.txt", delimiter = ",")
dataX = data[:, :2]
dataY = data[:, 2]
print(np.mean(dataX, axis = 1))
"""
