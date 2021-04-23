import numpy as np
from utils import *

np.random.seed(4)
base_path_results = 'Results/'
base_path_weights = 'Weights/'

class Model:
    
    def __init__(self, max_iter=100, alpha=0.01, reg=0, minibatch_size = 16, algo="minibatch", val_flag=True):
        self.W = None
        self.max_iter = max_iter
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.reg = reg
        self.record_cost = algo + '_' + str(max_iter) + '_' + str(alpha) + '_' + str(reg)
        self.record_evaluation_testing = algo + '_' + str(max_iter) + '_' + str(alpha) + '_' + str(reg) 
        self.record_evaluation_validation = algo + '_' + str(max_iter) + '_' + str(alpha) + '_' + str(reg) 
        if algo == 'minibatch':
            self.record_cost += '_' + str(minibatch_size)
            self.record_evaluation_testing += '_' + str(minibatch_size) 
            self.record_evaluation_validation += '_' + str(minibatch_size) 
        self.record_cost += '_' + 'cost_record.csv'
        self.record_evaluation_testing += '_' + 'testing_record.csv'
        self.record_evaluation_validation += '_' + 'val_record.csv'
        self.pred = None
        self.val_flag = val_flag
        self.J = 0
        self.algo = algo

        if not val_flag:
            self.record_cost = 'realtime_' + self.record_cost
            self.record_evaluation_testing = 'realtime_' + self.record_evaluation_testing

        self.record_cost = base_path_results + self.record_cost
        self.record_evaluation_testing = base_path_results + self.record_evaluation_testing
        self.record_evaluation_validation = base_path_results + self.record_evaluation_validation

    def save_weight(self, file_name="weight.npy", path=base_path_weights):
        file_name = str(self.max_iter) + '_' + str(self.alpha) + '_' + str(self.reg) + '_' + str(self.minibatch_size) + '_' + file_name
        if path != None:
            if path[-1] != '/':
                path += '/'
            np.save(path+file_name, self.W)
        else:
            np.save(file_name, self.W)

    def load_weight(self, file_name="weight.npy", path=base_path_weights):
        file_name = str(self.max_iter) + '_' + str(self.alpha) + '_' + str(self.reg) + '_' + str(self.minibatch_size) + '_' + file_name
        if path != None:
            if path[-1] != '/':
                path += '/'
            self.W = np.load(path+file_name)
        else:
            self.W = np.load(file_name)

    def cost(self, X, y):
        #print(X.shape, y.shape, self.W.shape)
        res = (1/X.shape[0])*np.sum(np.square(y - np.abs(np.dot(X, self.W)))) + self.reg * np.squeeze(np.dot(self.W.T, self.W))
        '''print(res)
        input()'''
        return res

    def gradient(self,X, y):
        #print(X.shape, y.shape, self.W.shape)
        #grad = (1 / X.shape[0]) * np.sum(-1*np.dot(X.T, y) + np.dot(np.dot(X.T, X), self.W) + self.reg * self.W)
        grad = (1 / X.shape[0]) * (np.dot(X.T, (y - np.abs(np.dot(X, self.W)))) + self.reg * self.W)
        return grad

    def update_weights(self, grad):
        self.W = self.W - self.alpha*(grad)

    def gradient_descent(self, X, y, train_file=None):
        self.J = self.cost(X, y)

        s = make_writable(self.W)
        train_file.write(s + ',' + str(self.J) + '\n')

        grad = self.gradient(X, y)
        self.update_weights(grad)
        
    def fit(self, X, y, test_X, test_y, val_X = None, val_y = None, mode = "normal", algo="minibatch"):
        self.W = np.random.rand(X.shape[1], 1)

        train = open(self.record_cost, 'w')
        test = open(self.record_evaluation_testing, 'w')
        val = None
        minibatches = None
        n = None
        
        train.write('W0,W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,W11,J\n')
        test.write('MAE,MSE,RMSE\n')
        
        if self.val_flag:
            val = open(self.record_evaluation_validation, 'w')
            val.write('MAE,MSE,RMSE\n')

        if self.algo == "minibatch":
            minibatches = createRandomMinibatches(X, y, self.minibatch_size)
            #n = len(minibatches)

        for i in range(self.max_iter):
            
            if self.algo == 'minibatch':
                #for j in range(n):
                 #   X, y = minibatches[j]
                for X, y in minibatches:
                    self.gradient_descent(X, y, train)
                    s = self.evaluate(test_X, test_y)
                
                    s = make_writable(s)
                    test.write(s + '\n')

            elif self.algo == 'batch':
                self.gradient_descent(X, y, train) 
                s = self.evaluate(test_X, test_y)
                
                s = make_writable(s)
                test.write(s + '\n')                   
                    
            if self.val_flag:
                s = self.evaluate(val_X, val_y)
                s = make_writable(s)
                val.write(s + '\n')

            if i % 10 == 0:
                print('Epoch No.:', i, 'Training Cost:',self.J)

        
        train.close()
        test.close()
        if self.val_flag:
            val.close()

    def predict(self, X):
        self.pred = np.abs(np.dot(X, self.W))

    def evaluate(self, X, y):
        self.predict(X)

        mae = np.sum(np.abs(y - self.pred)) / X.shape[0]
        mse = np.sum(np.square(y - self.pred)) / X.shape[0]
        rmse = np.sqrt(mse)

        return (mae, mse, rmse)

