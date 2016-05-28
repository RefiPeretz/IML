import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
#from builtins import range
from numpy.lib.polynomial import poly1d, polydiv
from array import *
import math

########## parameters #########
X = np.load("X_poly.npy")
Y = np.load("Y_poly.npy")
Xtrain = X[:100]
Ytrain = Y[:100]
Xvalid = X[100:200]
Yvalid = Y[100:200]
Xtest = X[200:]
Ytest = Y[200:]
###############################

def findH(h,x,y):
    list_h_erros = []
    for i in range(0,15):
        list_h_erros.append(calcError(h[i],x,y,100))
    return list_h_erros

    for g in range(0,15):
        print(list_h_erros[g])

def calcError(h,x,y,size_list):
    b = np.array(h)
    error= 0
    p = poly1d(b)
    for j in range(0,size_list):
        error += (math.pow((p(x[j]) - y[j]),2))
    error= error/size_list
    return error

######### training #############
h_d = []
for i in range(1, 16):
    h_d.append(np.polyfit(Xtrain,Ytrain,i))

###############################

######### validation ##########





############# k - fold cross validation algorithm ###########
def kFold(Xpart,Ypart):
    list_k_x = [Xpart[:40],Xpart[40:80],Xpart[80:120],Xpart[120:160],Xpart[160:200]]
    list_k_y = [Ypart[:40],Ypart[40:80],Ypart[80:120],Ypart[120:160],Ypart[160:200]]
    split_1_x = np.append(list_k_x[1],[list_k_x[2],list_k_x[3],list_k_x[4]])
    split_1_y = np.append(list_k_y[1],[list_k_y[2],list_k_y[3],list_k_y[4]])
    split_2_x = np.append(list_k_x[0],[list_k_x[2],list_k_x[3],list_k_x[4]])
    split_2_y = np.append(list_k_y[0],[list_k_y[2],list_k_y[3],list_k_y[4]])
    split_3_x = np.append(list_k_x[1],[list_k_x[0],list_k_x[3],list_k_x[4]])
    split_3_y = np.append(list_k_y[1],[list_k_y[0],list_k_y[3],list_k_y[4]])
    split_4_x = np.append(list_k_x[1],[list_k_x[2],list_k_x[0],list_k_x[4]])
    split_4_y = np.append(list_k_y[1],[list_k_y[2],list_k_y[0],list_k_y[4]])
    split_5_x = np.append(list_k_x[1],[list_k_x[2],list_k_x[3],list_k_x[0]])
    split_5_y = np.append(list_k_y[1],[list_k_y[2],list_k_y[3],list_k_y[0]])

    split_i_list_x = [split_1_x,split_2_x,split_3_x,split_4_x,split_5_x]
    split_i_list_y = [split_1_y,split_2_y,split_3_y,split_4_y,split_5_y]
    h_all_error_train = []
    h_all_error_validation = []
    for i in range(1,16):
        error_h_train = 0
        error_h_validation = 0
        for j in range(0,5):
            h = np.polyfit(split_i_list_x[j],split_i_list_y[j],i)
            error_h_train += calcError(h,Xpart,Ypart,200)
            error_h_validation += calcError(h,list_k_x[j],list_k_y[j],40)
        erro_h_train = error_h_train/5
        error_h_validation = error_h_validation/5
        print(erro_h_train,error_h_validation)
        h_all_error_train.append(erro_h_train)
        h_all_error_validation.append(error_h_validation)


    red_patch = mpatches.Patch(color='red', label='Training')
    b_patch = mpatches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch, b_patch])
    plt.xlabel('Degree')
    plt.ylabel("Error")
    plt.title("Least Squares k-fold k = 5")
    plt.plot(range(1,16),h_all_error_train,'r', label='Training')
    plt.plot(range(1,16),h_all_error_validation,'b', label="Validation")
    plt.show()

############# ploting ####################
list_erros_train = findH(h_d,Xtrain,Ytrain)
list_erros_valid = findH(h_d,Xvalid,Yvalid)
list_erros_test = findH(h_d,Xtest,Ytest)



red_patch = mpatches.Patch(color='red', label='Training')
b_patch = mpatches.Patch(color='blue', label='Testing')
g_patch = mpatches.Patch(color='green', label='Validation')
plt.legend(handles=[red_patch,b_patch,g_patch])
plt.xlabel('Degree')
plt.ylabel("Error")
plt.title("Least Squares algorithm train,test,validation")
plt.plot(range(1,16),list_erros_train,'r', label='Training')
plt.plot(range(1,16),list_erros_valid,'g', label="Validation")
plt.plot(range(1,16),list_erros_test,'b',label="Testing")
plt.show()

kFold(X[:200],Y[:200])
