"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: ID3

Author: Noga Zaslavsky
Date: April, 2016

"""
import numpy as np

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)


class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,gain = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        gain : the gain of splitting the data according to 'x[self.feature] < self.theta ?'
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.gain = gain
        self.label = label


class DecisionTree(object):
    """ A decision tree for bianry classification.
        Training method: ID3
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m,d = X.shape
        A = np.zeros(shape=(m, d))
        for j in range(d):
            sorted_X = sorted(X[:, j])
            for i in range(m-1):
                A[i,j] = (sorted_X[i] + sorted_X[i+1])/2
            A[m-1,j] = sorted_X[m-1]+0.5
        self.ID3(X, y, A, 0);
                

    def ID3(self,X, y, A, depth):
        """
        Gorw a decision tree with the ID3 recursive method

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        m,d = X.shape
        if depth < self.max_depth:
            if all(i == 1 for i in y):
                curr_node = Node(samples=y.size,label=1)
                if depth == 0:
                    self.root = curr_node
                return curr_node
            if all(i == -1 for i in y):
                curr_node = Node(samples=y.size,label=-1)
                if depth == 0:
                    self.root = curr_node
                return curr_node
            if y.size == 0:
                curr_node = Node(samples=y.size, label=(1 if np.sign(sum(y)) == 0 else np.sign(sum(y))))
                if depth == 0:
                    self.root = curr_node
                return curr_node
            else:
                if all((i == 0 or i == 1) for i in y):
                    curr_node = Node(samples=y.size,label=(1 if np.sign(sum(y)) == 0 else np.sign(sum(y))))
                    if depth == 0:
                        self.root = curr_node
                    return curr_node
                else:
                    gain_arr = self.info_gain(X,y,A)
                    x,j = np.unravel_index(gain_arr.argmax(), gain_arr.shape)

                    num_ones_j = sum(k < A[x, j] for k in X[:,j])
                    num_zeros_j = m - num_ones_j
                    one_feature = np.zeros(shape=(num_ones_j ,d))
                    zero_feature = np.zeros(shape=(num_zeros_j,d))
                    one_A = np.zeros(shape=(num_ones_j ,d))
                    zero_A = np.zeros(shape=(num_zeros_j ,d))
                    one_y = np.zeros(num_ones_j)
                    zero_y = np.zeros(num_zeros_j)
                    idx1 = 0
                    idx2 = 0
                    for i in range(m):
                        if X[i, j] < A[x, j]:
                            one_feature[idx1] = X[i]
                            one_A[idx1] = A[i]
                            one_y[idx1] = y[i]
                            idx1 += 1
                        else:
                            zero_feature[idx2] = X[i]
                            zero_A[idx2] = A[i]
                            zero_y[idx2] = y[i]
                            idx2 += 1
                    curr_node = Node(leaf=False,left=self.ID3(zero_feature,zero_y,zero_A,depth+1),
                                     right=self.ID3(one_feature,one_y,one_A,depth+1),
                                     samples=y.size,feature=j,theta=A[x,j],gain=gain_arr[x,j])
                    if depth == 0:
                        self.root = curr_node
                    return curr_node
        else:
            curr_node = Node(samples=y.size,label=(1 if np.sign(sum(y)) == 0 else np.sign(sum(y))))
            if depth == 0:
                self.root = curr_node
            return curr_node

    @staticmethod
    def info_gain(X, y, A):
        """
        Parameters
        ----------
        X, y : sample
        A : array of m*d real features, A[:,j] corresponds to thresholds over x_j

        Returns
        -------
        gain : m*d array containing the gain for each feature
        """
        m,d = A.shape
        binary_X = np.zeros(shape=(m,d))
        gain_arr = np.zeros(shape=(m,d))
        idx1 = 0
        for a in A:
            for i in range(m):
                for k in range(d):
                    if X[i,k] < a[k]:
                        binary_X[i,k] = 1
                    else: 
                        binary_X[i,k] = 0
            idx2 = 0
            for j in binary_X.T:
                p_y = sum(k == 1 for k in y)/y.size if y.size != 0 else 0
                p_xj = sum(k == 1 for k in j)/j.size if j.size != 0 else 0
                y_subset_ones = []
                y_subset_zeros = []
                for i in range(len(y)):
                    y_subset_ones.append(y[i]) if j[i] == 1 else y_subset_zeros.append(y[i])
                cond_p_y_xj_is_one = sum(k == 1 for k in y_subset_ones)/len(y_subset_ones) if len(y_subset_ones) != 0 else 0
                cond_p_y_xj_is_zero = sum(k == 1 for k in y_subset_zeros)/len(y_subset_zeros) if len(y_subset_zeros) != 0 else 0
                gain = entropy(p_y) - p_xj*entropy(cond_p_y_xj_is_one) - (1 - p_xj)*entropy(cond_p_y_xj_is_zero)
                gain_arr[idx1,idx2] = gain
                idx2 += 1
            idx1 += 1
        return gain_arr


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        m,d = X.shape
        y_hat = np.zeros(m)
        idx = 0
        for x in X:
            curr_node = self.root
            while not curr_node.leaf:
                if x[curr_node.feature] < curr_node.theta:
                    curr_node = curr_node.right
                else:
                    curr_node = curr_node.left
            y_hat[idx] = curr_node.label
            idx += 1
        return y_hat
            

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        return  np.mean(y_hat != y)
