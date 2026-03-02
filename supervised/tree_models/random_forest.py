import numpy as np
import os
import sys


from supervised.tree_models.decision_tree import DecisionTree
from supervised.tree_models.regression_tree import RegressionTree

class RandomForest:

    def __init__(self, n_estimators=10, max_depth=10,
                 min_samples_split=2, max_features=None,
                 task="classification"):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.trees = []


    def bootstrap_sample(self, X, y):

        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)

        return X[indices], y[indices]


    def fit(self, X, y):

        self.trees = []
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for a in range(self.n_estimators):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            if self.task == "classification":
                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion="gini"
                )
            else:
                tree = RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )

            tree.max_features = self.max_features
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def predict(self, X):

        tree_predictions = []

        for tree in self.trees:
            tree_predictions.append(tree.predict(X))

        tree_predictions = np.array(tree_predictions)

        final_predictions = []

        if self.task == "classification":

            for col in tree_predictions.T:
                values, counts = np.unique(col, return_counts=True)
                final_predictions.append(values[np.argmax(counts)])

        else:

            for col in tree_predictions.T:
                final_predictions.append(np.mean(col))

        return np.array(final_predictions)