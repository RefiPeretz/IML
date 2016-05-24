import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


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
h_d = [0]
for i in range(1, 16):
    h_d.append(np.polyfit(Xtrain,Ytrain,i))

for a in range(0,15):
    plt.plot(h_d[0])
    plt.show()

def findH(h,x,y):
    list_h_erros = []
    for i in range(0,15):
        list_h_erros.append(calcError(h[i],x,y,i+1))


def calcError(h,x,y,i):
    poly_list = [1]*i
    error = 0
    a = np.array(poly_list)
    b = np.array(h)
    for j in range(0,100):
        1+1



####### validation ########
def errorLs(hepo,x,y):
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