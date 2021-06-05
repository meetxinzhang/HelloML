from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model


dataPath = r"all.txt"
deliverData = genfromtxt(dataPath, delimiter=',')

print("data")
print(deliverData)

X = list(deliverData[:, 1:])
Y = list(deliverData[:, 0])

print("X:")
print(X)
print("Y")
print(Y)

regression = linear_model.LinearRegression()
regression.fit(X, Y)

print("coefficients: ")
print(regression.coef_)
print("intercept: ")
print(regression.intercept_)


xPred = [4.7, -0.92280543, 106.0699997,23,24000000,20000,1500,90,3416.06,2236.33,0.65465185,-1.34404022,33.51132466,-96045,702005]
xPred = np.array(xPred).reshape((1, -1))
yPred = regression.predict(xPred)
print("predicted y: ")
print(yPred)
