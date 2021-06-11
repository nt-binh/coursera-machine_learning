import numpy as np 
import matplotlib.pyplot as plt 
import math as m

def featureNormalize(arr):
    return (arr - np.mean(arr, axis = 0)) / np.std(arr, axis = 0)

def computeCost(dataX, dataY, theta):
    size = len(dataY)

    #hypothesis
    h = np.dot(dataX, theta)

    #cost function
    return (1 / (2 * size)) * np.sum(np.square(h - dataY))

def gradientDescent(dataX, dataY, theta, iter, alpha):
    size = len(dataY)
    J_history = []

    #start iterating with learning rate "alpha"
    for i in range(iter):
        #hypothesis
        h = np.dot(dataX, theta) 
        theta = theta - (alpha / size) * np.dot(h - dataY, dataX)
        J_history.append(computeCost(dataX, dataY, theta))
    return theta, J_history

def plotData(alphaGroup, dataX, dataY, theta, iter):
    plt.style.use("fivethirtyeight")
    #number of alpha
    numOfAlpha = len(alphaGroup)
    print(alphaGroup)
    #create subplot
    fig, ax = plt.subplots(nrows = m.ceil(numOfAlpha / 3), ncols = 3, sharex = True)
    row = 0
    col = 0
    for alpha in alphaGroup:
        theta, J_history = gradientDescent(dataX, dataY, theta, iter, alpha)
        if col < 3:
            ax[row, col].plot(range(iter), J_history)
            ax[row, col].set_title(f"Learning rate {alpha}")
            col+=1
            theta = np.zeros(3)
        else:
            row+=1
            col = 0
            ax[row, col].plot(range(iter), J_history)
            ax[row, col].set_title(f"Learning rate {alpha}")
            theta = np.zeros(3)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    data = np.genfromtxt("ex1data2.txt", delimiter = ",")
    dataX = data[:, :2]
    dataY = data[:, 2]
    size = len(dataY)   #length of dataX and dataY

    #feature normalization
    dataX = featureNormalize(dataX)
    dataY = featureNormalize(dataY)

    #add vector one's to dataX
    one = np.ones(size).reshape(size,1)
    dataX = np.concatenate((one, dataX), axis = 1)
    
    #initialize theta
    theta = np.zeros(3)
    print(f"With theta1 = {theta[0]}, theta2 = {theta[1]} & theta3 = {theta[2]}, computed Cost is {computeCost(dataX, dataY, theta)}")

    #gradient descent
    theta, J_history = gradientDescent(dataX, dataY, theta, 1500, 0.5)
    print(f"After gradient descent: ")
    for i in range(3):
        print(f"theta{i} : {theta[i]}")

    #choose suitable learning rate
    new_theta = np.zeros(3)
    plotData(np.linspace(0,1,6), dataX, dataY, new_theta, 300)
