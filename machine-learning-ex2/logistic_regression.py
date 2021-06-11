import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import optimize 
import utils 

def plotData(dataX, dataY):
    pos = (dataY == 1)
    neg = (dataY == 0)
    plt.plot(dataX[pos, 0], dataX[pos, 1], "r*", lw = 2, ms = 10)
    plt.plot(dataX[neg, 0], dataX[neg, 1], "bo")
    plt.tight_layout()
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend(["Admitted", "Not admitted"])
    

def sigmoid(dataX, theta):
    z = np.dot(dataX, theta)
    return 1 / (1 + np.exp(-z))

def costFunction(theta, dataX, dataY):
    size = len(dataY)
    #hypothesis
    h = sigmoid(dataX, theta)
    J = 0
    grad = np.zeros(len(theta))
    #return cost
    J = (1 / size) * np.sum(-dataY.dot(np.log(h)) - (1 - dataY).dot(np.log(1 - h)))
    grad = (1 / size) * (h - dataY).dot(dataX)
    return J, grad

def optimizeWithSCIPY(dataX, dataY, theta):
    #set options for optimize.minimize
    options = {"maxiter" : 100}

    #Scipy optimize.minimize function
    res = optimize.minimize(costFunction, 
                            theta,
                            (dataX, dataY),
                            jac = True,
                            method = "TNC",
                            options = options)
                            #TNC method is Truncated Newton method
                            #jac: method for computing gradient vector (only for CG, BFGS, Newton-CG)
                            #(dataX, dataY) is an extra argument passed to the mainfunction = costFunction
    cost = res.fun 
    grad = res.x
    print(f"Cost at theta found by optimize.minimize: {cost:.3f}")
    print(f"Theta : \n {grad}")
    return grad

def predict(theta, dataX):
    size = dataX.shape[0]
    p = np.zeros(size)
    p = np.round(sigmoid(dataX, theta))

    return p
    
if __name__ == "__main__":
    data = np.genfromtxt("ex2data1.txt", delimiter = ",")
    dataX = data[:, :2]
    dataY = data[:, 2]
    size = len(dataY)
    one = np.ones((size, 1))
    dataX = np.concatenate((one, dataX), axis = 1)
        
    #plotData(dataX, dataY)
    null_theta = np.zeros(3)
    test_theta = np.array([-24, .2, .2])
    cost1, grad1 = costFunction(null_theta, dataX, dataY)
    cost2, grad2 = costFunction(test_theta, dataX, dataY)
    print(f"Cost at null theta : {cost1:.3f}")
    print(f"Gradient at null theta : \n{grad1}\n")

    print(f"Cost at test theta : {cost2:.3f}")
    print(f"Gradient at test theta : \n{grad2}\n")

    #Optimize with scipy's optimize.minimize
    theta = optimizeWithSCIPY(dataX, dataY, null_theta)
    utils.plotDecisionBoundary(plotData, theta, dataX, dataY)
    plt.show()

    #Predict students with score 45 and 85
    prob = sigmoid([1, 45, 85], theta)
    print(f"For a student with scores 45 and 85, we predict an admission probability of {prob:.2f}")

    #Compute the training accuracy
    p = predict(theta, dataX)
    print(f"Train Accuracy is {np.mean(p == dataY) * 100}")
