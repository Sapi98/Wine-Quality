import numpy as np
import pandas as pd
from math import floor, ceil

def loadData(path):
    data = np.genfromtxt(path, delimiter=',')

    X = data[1:,:-1]
    y = np.reshape(data[1:,-1], (X.shape[0], 1))

    return (X, y)

def saveResults(path, result):
    pass

"""class Preprocess:

    def __init__(self):
        pass
"""
def featureNormalization(self, X):
    mean = X.mean()
    sd = X.std()
    X = X - mean / sd

    return X

def shuffle(X, y):
    p = np.random.permutation(y.size)

    X = X[p,:]
    y = np.reshape(y[p], (y.size, 1))
    
    return X, y

def splitData(X, y, train=0.7, test=0.15):
    X, y = shuffle(X, y)

    n = X.shape[0]

    train_X = X[:floor(train*n)]
    train_Y = y[:floor(train*n)]

    test_X = X[floor(train * n):]
    test_Y = y[floor(train * n):]

    return (train_X, train_Y, test_X, test_Y)

def createRandomMinibatches(self, X, y, minibatch_size=16):
    minibatches = []
    
    n = y.size // minibatch_size
    
    X, y = shuffle(X, y)

    for i in range(n):
        minibatch = (X[i*minibatch_size:(i+1)*minibatch_size, :], y[i*minibatch_size:(i+1)*minibatch_size, :])
        minibatches.append(minibatch)

    minibatches.append(X[n*minibatch_size:, :], y[n*minibatch_size:, :])

    return minibatches

if __name__ == "__main__":
    x = [[1,2,3],[4,5,6],[7,8,9]]
    x = np.array(x)
    y=np.array([1,2,3])

    shuffle(x,y)