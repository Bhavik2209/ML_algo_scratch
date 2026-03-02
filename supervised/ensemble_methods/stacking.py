import numpy as np


class Stacking:

    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):

        for model in self.base_models:
            model.fit(X, y)

        meta_features = []

        for model in self.base_models:
            preds = model.predict(X)
            meta_features.append(preds)

        meta_X = np.array(meta_features).T

        self.meta_model.fit(meta_X, y)

    def predict(self, X):

        meta_features = []

        for model in self.base_models:
            preds = model.predict(X)
            meta_features.append(preds)

        meta_X = np.array(meta_features).T

        return self.meta_model.predict(meta_X)