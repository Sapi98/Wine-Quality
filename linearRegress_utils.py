import numpy as np
from utils import *

class Model:
    
    def __init__(self, max_iter=100, alpha=0.01, reg=0, minibatch_size = 16):
        self.W = None
        self.max_iter = max_iter
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.reg = reg
        self.record_cost = []
        self.pred = None
        self.record_evaluation_testing = []
        self.record_evaluation_validation = []

    def save_weight(self, file_name="weight.npy", path=None):
        if path != None:
            np.save(path, self.W)
        else:
            np.save(file_name, self.W)

    def load_weight(self, file_name="weight.npy", path=None):
        if path != None:
            self.W = np.load(path)
        else:
            self.W = np.load(file_name)

    def cost(self, X, y):
        res = (1/X.shape[0])*np.sum(np.square(y - np.dot(self.W.T, X))) + self.reg * np.dot(self.W.T, self.W)
        return res

    def gradient(self,X, y):
        grad = (2 / X.shape[0]) * np.sum(-1*np.dot(X.T, y) + np.dot(X.T, X) * self.W) + 2 * self.reg * self.W
        return grad

    def update_weights(self, grad):
        self.W = self.W - self.alpha*(grad)
        
    def fit(self, X, y, test_X, test_y, algo="minibatch"):
        self.W = np.random.rand((X.shape[0], 1))

        if algo == "minibatch":
            minibatches = createRandomMinibatches(X, y, self.minibatch_size)
            n = len(minibatches)
            j = 0
            for _ in range(self.max_iter):
                
                for i in range(n):
                    X, y = minibatches[i]
                    J = self.cost(X, y)
                    grad = self.gradient(X, y)
                    self.update_weights(grad)

                    self.record_cost.append(J)

                    self.record_evaluation_validation.append(self.evaluate(val_X, val_y))
                    self.record_evaluation_testing.append(self.evaluate(test_X, test_y))

    def predict(self, X):
        self.pred = np.dot(self.W.T, X)

    def evaluate(self, X, y):
        self.predict(X)

        mae = np.sum(np.abs(y - self.pred)) / X.shape[0]
        mse = np.sum(np.square(y - self.pred)) / X.shape[0]
        rmse = np.sqrt(mse)

        return (mae, mse, rmse)