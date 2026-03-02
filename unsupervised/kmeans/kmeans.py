import numpy as np

class KMeans:

    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape


        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iterations):


            clusters = self.assign_clusters(X)

            new_centroids = self.calculate_centroids(X, clusters)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def assign_clusters(self, X):
        clusters = []

        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            cluster_index = np.argmin(distances)
            clusters.append(cluster_index)

        return np.array(clusters)

    def calculate_centroids(self, X, clusters):
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))

        for i in range(self.k):
            points = X[clusters == i]
            centroids[i] = np.mean(points, axis=0)

        return centroids

    def predict(self, X):
        return self.assign_clusters(X)