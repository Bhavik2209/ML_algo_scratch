import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:

    def __init__(self, max_depth=10, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None


    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)


    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        entropy_value = 0
        for p in probabilities:
            entropy_value += -p * np.log2(p)

        return entropy_value


    def impurity(self, y):
        if self.criterion == "entropy":
            return self.entropy(y)
        else:
            return self.gini(y)

    def best_split(self, X, y):

        best_score = -1   # we maximize impurity reduction
        best_feature = None
        best_threshold = None

        rows, cols = X.shape
        parent_impurity = self.impurity(y)

        for col in range(cols):

            values = np.unique(X[:, col])

            for threshold in values:

                left_indices = []
                right_indices = []

                for i in range(rows):
                    if X[i, col] <= threshold:
                        left_indices.append(i)
                    else:
                        right_indices.append(i)

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]

                left_impurity = self.impurity(y_left)
                right_impurity = self.impurity(y_right)

                weighted_impurity = (
                    (len(y_left) / rows) * left_impurity +
                    (len(y_right) / rows) * right_impurity
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_score:
                    best_score = gain
                    best_feature = col
                    best_threshold = threshold

        return best_feature, best_threshold



    def most_common(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]


    def build(self, X, y, depth=0):

        if (
            depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1
        ):
            return Node(value=self.most_common(y))

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=self.most_common(y))

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