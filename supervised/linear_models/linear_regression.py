import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters =  n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            errors = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model is not trained yet. Please call fit() before predict().")
        return np.dot(X, self.weights) + self.bias