import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        std = np.where(self.std == 0, 1, self.std)
        return (X - self.mean) / std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)