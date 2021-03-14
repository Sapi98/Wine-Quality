import numpy as np
from utils import *

class Model:
    
    def __init__(self, max_iter=100, alpha=0.01, reg=0, minibatch_size = 16, val_flag=True):
        self.W = None
        self.max_iter = max_iter
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.reg = reg
        self.record_cost = str(max_iter) + '_' + str(alpha) + '_' + str(reg) + '_' + str(minibatch_size) + '_' + 'cost_record.txt'
        self.pred = None
        self.record_evaluation_testing = str(max_iter) + '_' + str(alpha) + '_' + str(reg) + '_' + str(minibatch_size) + '_' + 'testing_record.txt'
        self.record_evaluation_validation = str(max_iter) + '_' + str(alpha) + '_' + str(reg) + '_' + str(minibatch_size) + '_' + 'val_record.txt'
        self.val_flag = val_flag

        if not val_flag:
            self.record_cost = 'realtime_' + self.record_cost
            self.record_evaluation_testing = 'realtime_' + self.record_evaluation_testing

    def save_weight(self, file_name="weight.npy", path=None):
        if path != None:
            np.save(path+'/'+file_name, self.W)
        else:
            np.save(file_name, self.W)

    def load_weight(self, file_name="weight.npy", path=None):
        if path != None:
            self.W = np.load(path+'/'+file_name)
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
        
    def fit(self, X, y, test_X, test_y, val_X = None, val_y = None, mode = "normal", algo="minibatch"):
        self.W = np.random.rand((X.shape[0], 1))

        train = open(self.record_cost, 'w')
        test = open(self.record_evaluation_testing, 'w')
        
        train.write('W0,W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,J\n')
        test.write('MAE,MSE,RMSE\n')
        
        if self.val_flag:
            val = open(self.record_evaluation_validation, 'w')
            val.write('MAE,MSE,RMSE\n')

        if algo == "minibatch":
            minibatches = createRandomMinibatches(X, y, self.minibatch_size)
            n = len(minibatches)

            for i in range(self.max_iter):
                
                for j in range(n):
                    X, y = minibatches[j]
                    J = self.cost(X, y)

                    s = make_writable(self.W)
                    train.write(s + ',' + str(J) + '\n')

                    grad = self.gradient(X, y)
                    self.update_weights(grad)

                    if self.val_flag:
                        s = self.evaluate(val_X, val_y)
                        s = make_writable(s)
                        val.write(s + '\n')
                    
                    s = self.evaluate(test_X, test_y)
                    
                    if i % 20 == 0 and j % 10 == 0:
                        print('Iter No.:', i, 'Minibatch No.:', j, 'Training Cost:',J,'')

                    s = make_writable(s)
                    test.write(s + '\n')

                    
        
        train.close()
        test.close()
        if self.val_flag:
            val.close()

    def predict(self, X):
        self.pred = np.dot(self.W.T, X)

    def evaluate(self, X, y):
        self.predict(X)

        mae = np.sum(np.abs(y - self.pred)) / X.shape[0]
        mse = np.sum(np.square(y - self.pred)) / X.shape[0]
        rmse = np.sqrt(mse)

        return (mae, mse, rmse)

