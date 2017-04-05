# -*- coding: utf-8 -*-
"""
@author: Refi
"""

import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m,d = X.shape
        D = np.ones(m)/m
        epsilon = 0
        denominator = 0
        indicator = 0
        for t in range(self.T):
            self.h[t] = self.WL(D,X,y)
            for i in range(m):
                if (self.h[t].predict(X))[i] != y[i]:
                    indicator = 1
                else:
                    indicator = 0
                epsilon += D[i]*indicator
            self.w[t] = 0.5*np.log(1/epsilon - 1)
            for k in range(m):
                denominator += (D[k]*np.e**(-self.w[t]*y[k]*(self.h[t].predict(X))[k]))
            for j in range(m):
                D[j] = (D[j]*np.e**(-self.w[t]*y[j]*(self.h[t].predict(X))[j]))/denominator
            denominator = 0
            epsilon = 0

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        sum0 = 0
        for i in range(self.T):
            sum0 += self.w[i]*self.h[i].predict(X)
        return np.sign(sum0)


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        return  np.mean(y_hat != y)
