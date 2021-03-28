import numpy as np
import pandas as pd
from math import floor, ceil
import pandas as pd

def loadData(path):
    #data = np.genfromtxt(path, delimiter=',')
    data = pd.read_csv(path)

    return data

def saveResults(path, result):
    pass

"""class Preprocess:

    def __init__(self):
        pass
"""
def featureNormalization(X, epsilon=1e-8):
    mean = X.mean(axis=0)
    sd = X.std(axis=0) + epsilon
    X = (X - mean) / sd

    return X

def shuffle(X, y=None):
    p = np.random.permutation(X.shape[0])

    X = X[p,:]
    
    if type(y) != type(None):
        y = np.reshape(y[p], (X.shape[0], 1))
    
    return X, y

def splitData(data, y=None, train=0.7):
    data_array = np.array(data)[1:]
    b = np.ones((data_array.shape[0], 1))
    data_array = np.concatenate((b,data_array), axis=1)
    #print(data_array)
    train_data = None
    
    X, y = shuffle(data_array, y)

    n = X.shape[0]

    if type(y) != type(None):
        train_X = X[:floor(train*n)]
        train_Y = np.reshape(y[:floor(train*n)], (train_X.shape[0], 1))

        test_X = X[floor(train * n):]
        test_Y = np.reshape(y[floor(train * n):], (test_X.shape[0], 1))

    else:
        train_X = data_array[:floor(train*n),:-1]
        train_X = np.concatenate((np.ones((train_X.shape[0], 1)), train_X), axis=1)
        train_Y = np.reshape(data_array[:floor(train*n),-1], (train_X.shape[0], 1))

        test_X = data_array[floor(train * n):, :-1]
        test_X = np.concatenate((np.ones((test_X.shape[0], 1)), test_X), axis=1)
        test_Y = np.reshape(data_array[floor(train * n):, -1], (test_X.shape[0], 1))

        train_data = pd.DataFrame(data_array, columns=data.columns)

    return (train_data, train_X, train_Y, test_X, test_Y)

def createRandomMinibatches(X, y, minibatch_size=16):
    minibatches = []
    
    n = y.size // minibatch_size
    
    X, y = shuffle(X, y)

    for i in range(n):
        minibatch = (X[i*minibatch_size:(i+1)*minibatch_size, :], y[i*minibatch_size:(i+1)*minibatch_size, :])
        minibatches.append(minibatch)

    minibatches.append(X[n*minibatch_size:, :], y[n*minibatch_size:, :])

    return minibatches

def make_writable(data):
    s = str(data[0])

    for i in range(1, len(data)):
        s += ',' + str(data[i])
    
    return s

def generateValidation(X, y, val=0.15):
    X, y = shuffle(X, y)

    n = X.shape[0]

    train_X = X[:floor((1-val)*n)]
    train_y = np.reshape(y[:floor((1-val)*n)], (train_X.shape[0], 1))

    val_X = X[floor((1-val) * n):]
    val_y = np.reshape(y[floor((1-val) * n):], (val_X.shape[0], 1))

    return (train_X, train_y, val_X, val_y)

if __name__ == "__main__":
    x = [[1,2,3],[4,5,6],[7,8,9]]
    x = np.array(x)
    y=np.array([1,2,3])

    shuffle(x,y)