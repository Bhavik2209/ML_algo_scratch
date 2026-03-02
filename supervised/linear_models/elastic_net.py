import numpy as np


class ElasticNetRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_l1=1.0, lambda_l2=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):

            y_pred = np.dot(X, self.weights) + self.bias


            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            dw = dw + self.lambda_l1 * np.sign(self.weights)

            dw = dw + 2 * self.lambda_l2 * self.weights

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias