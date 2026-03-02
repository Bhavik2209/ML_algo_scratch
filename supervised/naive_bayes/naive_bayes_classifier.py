import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i in range(n_classes):
            c = self.classes[i]

            X_c = X[y == c]

            self.mean[i] = np.mean(X_c, axis=0)
            self.var[i] = np.var(X_c, axis=0) + 1e-9
            self.priors[i] = len(X_c) / n_samples


    def predict(self, X):
        predictions = []

        for x in X:
            predictions.append(self.predict_single(x))

        return np.array(predictions)


    def predict_single(self, x):
        posteriors = []

        for i in range(len(self.classes)):

            prior = np.log(self.priors[i])

            # we are taking log here because of the product of probabilities, which can lead to underflow. By taking log, we convert the product into a sum, which is more stable.
            likelihood = np.sum(
                np.log(self.gaussian_pdf(i, x))
            )

            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]


    def gaussian_pdf(self, class_index, x):

        mean = self.mean[class_index]
        var = self.var[class_index]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator