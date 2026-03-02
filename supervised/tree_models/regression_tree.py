import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class RegressionTree:

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None


    def mse(self, y):
        if len(y) == 0:
            return 0
        mean_value = np.mean(y)
        return np.mean((y - mean_value) ** 2)


    def best_split(self, X, y):

        best_score = float("inf")
        best_feature = None
        best_threshold = None

        rows, cols = X.shape

        for col in range(cols):

            values = np.unique(X[:, col])
            values = np.sort(values)


            for i in range(len(values) - 1):

                threshold = (values[i] + values[i + 1]) / 2 #midpoints

                left_indices = []
                right_indices = []

                for r in range(rows):
                    if X[r, col] <= threshold:
                        left_indices.append(r)
                    else:
                        right_indices.append(r)

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]

                mse_left = self.mse(y_left)
                mse_right = self.mse(y_right)

                weighted_mse = (
                    (len(y_left) / rows) * mse_left +
                    (len(y_right) / rows) * mse_right
                )

                if weighted_mse < best_score:
                    best_score = weighted_mse
                    best_feature = col
                    best_threshold = threshold

        return best_feature, best_threshold



    def build(self, X, y, depth=0):

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.mean(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=np.mean(y))

        left_X = []
        left_y = []
        right_X = []
        right_y = []

        for i in range(len(X)):
            if X[i, feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        left_X = np.array(left_X)
        left_y = np.array(left_y)
        right_X = np.array(right_X)
        right_y = np.array(right_y)

        left_child = self.build(left_X, left_y, depth + 1)
        right_child = self.build(right_X, right_y, depth + 1)

        return Node(feature, threshold, left_child, right_child)


    def fit(self, X, y):
        self.root = self.build(X, y)


    def predict_one(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)


    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_one(x, self.root))
        return np.array(predictions)