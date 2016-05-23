import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from  sklearn import cross_validation
#from builtins import range

from array import *


X = np.load("X_poly.npy")
Y = np.load("Y_poly.npy")
Xtrain = X[:99]
Ytrain = Y[:99]
Xvalid = X[100:199]
Yvalid = Y[100:199]
Xtest = X[200:]
Ytest = Y[200:]

####### training #########
h_d = []
for i in range(0, 15):
    h_d.append(np.polyfit(Xtrain,Ytrain,i))

for a in range(0,15):
    plt.plot(h_d[a])
    plt.show()

####### validation ########
def errorLs(hepoteza,x,y):
    LS = 0;

    LS = 0.0025 + 1
    LS = LS/100
    return LS

h_loss = []
for j in range(0, 15):
    h_loss.append(errorLs(h_d[j],Xtrain,Ytrain))


######### training #############
## todo - choose hepoteza hfinal
hfinal = 1
lastError = errorLs(hfinal,Xtest,Ytest)





############# k - fold cross validation algorithm ###########
k = 5
#score = cross_validation(k);
#Xtrain = X[:99]
#Ytrain = Y[:99]
#Xvalid = X[100:199]
#Yvalid = Y[100:199]



####### random ##########
# np.random.uniform(0,1,1000)
# np.random.normal(mu,sigma,1000)

####### operations #######
# np.linalg.pinv(A)
# np.column_stack((u,v))
# np.concatenate((a,b))


######## ploting ##########
#plt.plot(X)
#plt.scatter(X,Y)
# plt.show()
#print(Xtrain)
#print(Ytrain)