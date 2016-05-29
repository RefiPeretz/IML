# -*- coding: utf-8 -*-
"""
@author: Refi
"""
import numpy as np
from ex6_tools import DecisionStump, decision_boundaries, view_dtree
from adaboost import AdaBoost
from nearest_neighbors import kNN
from decision_tree import DecisionTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


X_train = np.loadtxt('SynData\X_train.txt')
y_train = np.loadtxt('SynData\y_train.txt')
X_val = np.loadtxt('SynData\X_val.txt')
y_val = np.loadtxt('SynData\y_val.txt')
X_test = np.loadtxt('SynData\X_test.txt')
y_test = np.loadtxt('SynData\y_test.txt')


def Q3():  # AdaBoost
    print("Q3")
    print("===============================================")
    T = [None]*41
    T[0] = 1
    for i in range(5, 201, 5):
        T[i//5] = i
    
    classifiers = [None]*41
    train_err = [None]*41
    val_err = [None]*41
    for i in range(len(T)):
        classifiers[i] = AdaBoost(DecisionStump, T[i])
        classifiers[i].train(X_train, y_train)
        train_err[i] = classifiers[i].error(X_train, y_train)
        val_err[i] = classifiers[i].error(X_val, y_val)
    
    plt.figure(1)
    plt.subplot(3, 2, 1)
    decision_boundaries(classifiers[0], X_train, y_train, "Training Classification T=1")
    plt.subplot(3, 2, 2)
    decision_boundaries(classifiers[1], X_train, y_train, "Training Classification T=5")
    plt.subplot(3, 2, 3)
    decision_boundaries(classifiers[2], X_train, y_train, "Training Classification T=10")
    plt.subplot(3, 2, 4)
    decision_boundaries(classifiers[10], X_train, y_train, "Training Classification T=50")
    plt.subplot(3, 2, 5)
    decision_boundaries(classifiers[20], X_train, y_train, "Training Classification T=100")
    plt.subplot(3, 2, 6)
    decision_boundaries(classifiers[40], X_train, y_train, "Training Classification T=200")

    plt.show()
    plt.figure(2)
    red_patch = mpatches.Patch(color='red', label='Training')
    b_patch = mpatches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch, b_patch])
    plt.plot(T, train_err, 'r', T, val_err, 'b')
    plt.title("Training Error and Validation Error ")
    
    plt.show()
    
    T_hat = T[np.argmin(val_err)]
    print("the value of T_hat (T that minimize validation error) is:", T_hat) #55
    print("the test error of T_hat is:", classifiers[T_hat//5].error(X_test, y_test)) #0.184
    plt.figure(3)
    decision_boundaries(classifiers[T_hat//5], X_train, y_train, "Training Classification of T_hat")
    plt.show()
    print("===============================================")
    return

def Q4():  # decision trees
    print("Q4")
    print("===============================================")
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    classifiers = [None]*12
    train_err = [None]*12
    val_err = [None]*12
    for i in range(len(max_depth)):
        classifiers[i] = DecisionTree(max_depth[i])
        classifiers[i].train(X_train,y_train)
        train_err[i] = classifiers[i].error(X_train,y_train)
        val_err[i] = classifiers[i].error(X_val,y_val)
    
    plt.figure(1)
    for j in range(len(max_depth)):
        plt.subplot(4, 3, j+1)
        msg = "Training Classification depth="+str(max_depth[j])
        decision_boundaries(classifiers[j],X_train,y_train,msg)
    
    plt.figure(2)
    plt.plot(max_depth, train_err, 'r', max_depth, val_err, 'b')
    plt.title("Training Error (red) and Validation Error (blue)")
    plt.show()
    
    max_depth_hat = max_depth[np.argmin(val_err)]
    print("the value of max_depth_hat (max_depth that minimize validation error) is:", max_depth_hat) #4
    print("the test error of max_depth_hat is:", classifiers[max_depth.index(max_depth_hat)].error(X_test,y_test)) #0.11

    view_dtree(classifiers[max_depth.index(max_depth_hat)])
    print("===============================================")
    
    return

def Q5():  # kNN
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


votes = np.loadtxt('CongressData\\votes.txt')
parties = np.loadtxt('CongressData\\parties.txt')
feature_names = np.loadtxt('CongressData\\feature_names.txt', dtype=bytes).astype(str)
class_names = np.loadtxt('CongressData\\class_names.txt', dtype=bytes).astype(str)

def Q6(): # Republican or Democrat?
    print("Q6")
    print("===============================================")
    votes_tmp = np.column_stack((votes, parties))
    training_votes, val_votes, test_votes = np.vsplit(votes_tmp[np.random.permutation(votes_tmp.shape[0])],(217,391))
    training_parties = training_votes[:, 16]
    training_votes = np.delete(training_votes, np.s_[16:17], axis=1)
    val_parties = val_votes[:, 16]
    val_votes = np.delete(val_votes, np.s_[16:17], axis=1)
    test_parties = test_votes[:, 16]
    test_votes = np.delete(test_votes, np.s_[16:17], axis=1)
    adaboost_classifiers = [None]*5
    dtree_classifiers = [None]*5
    knn_classifiers = [None]*5
    adaboost_val_err = [None]*5
    dtree_val_err = [None]*5
    knn_val_err = [None]*5
    T = [1, 25, 50, 100, 200]
    k = [1, 5, 25, 100, 200]
    d = [1, 5, 10, 16, 20]

    for i in range(5):
        dtree_classifiers[i] = DecisionTree(d[i])
        dtree_classifiers[i].train(training_votes, training_parties)
        dtree_val_err[i] = dtree_classifiers[i].error(val_votes, val_parties)
        adaboost_classifiers[i] = AdaBoost(DecisionStump, T[i])
        adaboost_classifiers[i].train(training_votes, training_parties)
        adaboost_val_err[i] = adaboost_classifiers[i].error(val_votes, val_parties)
        knn_classifiers[i] = kNN(k[i])
        knn_classifiers[i].train(training_votes, training_parties)
        knn_val_err[i] = knn_classifiers[i].error(val_votes, val_parties)

    """
    explanation for choosing the parameters for each classifier: I trained some classifiers of each type
    with different parameters and then measured the validation error with the validation sample.
    then,as I did in previous tasks, I chose the parameter that minimize the validation error over
    the sample and used the classifiers with this parameter to measure the test error.
    here is plots with validation error of each classifier over some parameters:
    """
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(d, dtree_val_err)
    plt.title("Validation Error of Decision Tree")
    plt.subplot(3, 1, 2)
    plt.plot(T, adaboost_val_err)
    plt.title("Validation Error of Adaboost")
    plt.subplot(3, 1, 3)
    plt.plot(k, knn_val_err)
    plt.title("Validation Error of k Nearest Neighbors")
    plt.show()

    d_hat = d[np.argmin(dtree_val_err)]
    T_hat = T[np.argmin(adaboost_val_err)]
    k_hat = k[np.argmin(knn_val_err)]
    print("Decision Tree: the optimal validation error is: ", dtree_val_err[d.index(d_hat)],
          " , and the optimal test error is: ", dtree_classifiers[d.index(d_hat)].error(test_votes, test_parties))
    print("Adaboost: the optimal validation error is: ", adaboost_val_err[T.index(T_hat)],
          " , and the optimal test error is: ", adaboost_classifiers[T.index(T_hat)].error(test_votes, test_parties))
    print("k Nearest Neighbors: the optimal validation error is: ", knn_val_err[k.index(k_hat)],
          " , and the optimal test error is: ", knn_classifiers[k.index(k_hat)].error(test_votes, test_parties))

    #optional

    dtree1 = DecisionTree(3)
    dtree1.train(votes[:10, :], parties[:10])
    view_dtree(dtree1, feature_names, class_names)
    dtree2 = DecisionTree(6)
    dtree2.train(votes[:150, :], parties[:150])
    view_dtree(dtree2, feature_names, class_names)
    dtree3 = DecisionTree(10)
    dtree3.train(votes, parties)
    view_dtree(dtree3, feature_names, class_names)

    print("===============================================")

    return

if __name__ == '__main__':
    
    #Q3()
    Q4()
    Q5()
    Q6()
    pass