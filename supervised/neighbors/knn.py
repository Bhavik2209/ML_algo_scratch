import numpy as np

class KNearestNeighbors:

    def __init__(self, k=5, distance_metric='euclidean', task='classification'):
        self.k = k
        self.distance_metric = distance_metric
        self.task = task

    def distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            return np.sum(np.abs(x1 - x2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        distances = []

        for x_train in self.X_train:
            d = self.distance(x, x_train)
            distances.append(d)

        distances = np.array(distances)

        k_indices = np.argsort(distances)[:self.k]
        k_values = self.y_train[k_indices]

        if self.task == 'classification':
            values, counts = np.unique(k_values, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(k_values)

    def predict(self, X):
        predictions = []

        for x in X:
            pred = self.predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)

        if self.task == 'classification':
            return np.mean(predictions == y)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)