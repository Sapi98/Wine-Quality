import numpy as np

class Model:
    
    def __init__(self, max_iter=100, alpha=0.01, l=0):
        self.W = None
        self.max_iter = max_iter
        self.alpha = alpha
        self.l = l

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
        res = (1/X.shape[0])*np.sum(np.square(y - np.dot(self.W.T, X))) + self.l*np.dot(self.W.T, self.W)
        return res

    def gradient(self,X, y):
        grad = J*X
        return grad

    def update_weights(self, grad):
        self.W = self.W - self.alpha*(grad)
        
    def fit(self, X, y):
        self.W = np.random((X.shape[1], 1))

        for _ in range(self.max_iter):
            J = self.cost(X, y)
            grad = self.gradient(X, y)
            self.update_weights(grad)

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    