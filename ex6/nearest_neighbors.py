"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the k nearest neighbors classifier.

Author: Noga Zaslavsky
Date: April, 2016

"""

import numpy as np
from numpy import matlib
from ex6_tools import decision_boundaries
import matplotlib.pyplot as plt

class kNN(object):
    def __init__(self, k):
        self.K = k
        self.Xtrain = None
        self.Ytrain = None
        return

    def train(self, X, y):
        self.Xtrain = X
        self.Ytrain = y


    def calcLabel(self,labels):
        sumOflabels = sum(labels)
        if(sumOflabels > 0 ):
            return 1
        return -1

    def distCalc(self,x,XList):
        return np.linalg.norm(x-XList)

    def predictForOne(self,x):

        y_sorted = self.Ytrain[np.argsort(np.linalg.norm(self.Xtrain - x,axis=1))]
        relevantDist = y_sorted[:self.K]
        #for index in relevantDistIndex:
            #listOfLables.append(self.Ytrain[index])

        return self.calcLabel(relevantDist)


    def predict(self, X):
        y_hat = []
        i = 0
        y_hat = np.array([])

        for x in X:
            print(i)
            y_hat = np.append(y_hat,self.predictForOne(x))
            i+=1
        print(y_hat)
        return y_hat


        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        # TODO - implement this method

    def calcError(self,y_predict,y):
        print(y.size,y_predict.size)
        sumError = 0;
        for i in range(0,y.size):
            if y_predict[i] != y[i]:
                sumError += 1
        return sumError/y.size

    def error(self, X, y):

        y_predict = self.predict(X)
        return self.calcError(y_predict,y)
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        # TODO - implement this method


trainX = np.loadtxt("X_train.txt")
trainY = np.loadtxt("y_train.txt")
testX = np.loadtxt("X_val.txt")
testY = np.loadtxt("y_val.txt")

k_list = [1,3,10,100,200,500]
error_train =[]
error_valid =[]
for k in k_list:
    papo = kNN(k)
    papo.train(trainX, trainY)
    error_train.append(papo.error(trainX,trainY))
    error_valid.append(papo.error(testX, testY))



#print(error_valid)

plt.xlabel("samples")
plt.ylabel("error")
plt.title("Knn algorithem - validation and training error")
plt.plot(k_list,error_train,'r')
plt.plot(k_list,error_valid,'g')
plt.show()

for a in k_list:
    papo1 = kNN(a)
    papo1.train(trainX, trainY)
    decision_boundaries(papo1,trainX,trainY,'kNN decision boundaries and training data, K = '+str(a))