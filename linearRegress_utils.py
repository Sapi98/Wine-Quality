import numpy as np

class Model:
    
    def __init__(self, max_iter=100, alpha=0.01):
        self.W = None
        self.max_iter = max_iter
        self.alpha = alpha

    def save_weight(self, file_name="weight.npy", path=None):
        np.save()

    def load_weight(self, file_name="weight.npy", path=None):
        pass

    def cost(self, X, y):
        pass

    def gradient(self, X, y):
        pass

    def update_weights(self, grad):
        pass

    def fit(self, X, y):
        w = self
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    