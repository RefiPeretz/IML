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
        dup = matlib.repmat(x, self.Xtrain.size/2,1 )
        #dist = np.array([])
        #dist = np.linalg.norm(x-self.Xtrain)
        #for point in self.Xtrain:

            #dist = np.append(dist,np.linalg.norm(x-point))
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


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        # TODO - implement this method



trainX = np.loadtxt("X_train.txt")
trainY = np.loadtxt("y_train.txt")
testX = np.loadtxt("X_train.txt")
testY = np.loadtxt("y_train.txt")

papo = kNN(10)

papo.train(trainX,trainY)

decision_boundaries(papo,testX,testY,"title")
