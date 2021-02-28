import numpy as np

class Model:
    
    def __init__(self, max_iter):
        self.W = None
        self.max_iter = max_iter

    def save_weight(self, file_name="weight.npy", path=None):
        np.save()

    def load_weight(self, file_name="weight.npy", path=None):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    