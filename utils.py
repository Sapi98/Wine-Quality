import numpy as np
import pandas as pd
from math import floor, ceil

def loadData(path):
    data = np.genfromtxt(path, delimiter=',')

    X = data[1:,:-1]
    y = np.reshape(data[1:,-1], (X.shape[0], 1))

    return (X.T, y)

def saveResults(path, result):
    pass

"""class Preprocess:

    def __init__(self):
        pass
"""
def featureNormalization(self):
    pass

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

    test_X = X[floor((train+val)*n):]
    test_Y = y[floor((train+val)*n):]

    return (train_X, train_Y, test_X, test_Y)

def createRandomMinibatches(self):
    pass


if __name__ == "__main__":
    x = [[1,2,3],[4,5,6],[7,8,9]]
    x = np.array(x)
    y=np.array([1,2,3])

    shuffle(x,y)