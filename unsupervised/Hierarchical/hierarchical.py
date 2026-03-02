import numpy as np


class HierarchicalClustering:

    def __init__(self, n_clusters=2, linkage='average', method='agglomerative'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.method = method
        self.labels = None

    def fit(self, X):

        if self.method == 'agglomerative':
            self.fit_agglomerative(X)
        else:
            self.fit_divisive(X)


    def fit_agglomerative(self, X):

        n_samples = X.shape[0]

        clusters = []
        for i in range(n_samples):
            clusters.append([i])

        while len(clusters) > self.n_clusters:

            min_distance = float('inf')
            cluster_pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):

                    distance = self.calculate_cluster_distance(
                        X, clusters[i], clusters[j]
                    )

                    if distance < min_distance:
                        min_distance = distance
                        cluster_pair = (i, j)

            i, j = cluster_pair
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)

        self.assign_labels(clusters, n_samples)


    def fit_divisive(self, X):

        n_samples = X.shape[0]


        clusters = [list(range(n_samples))]

        while len(clusters) < self.n_clusters:


            max_distance = -1
            cluster_to_split = None

            for cluster in clusters:
                distance = self.cluster_spread(X, cluster)
                if distance > max_distance:
                    max_distance = distance
                    cluster_to_split = cluster

            clusters.remove(cluster_to_split)

            # Split using simple 2-means logic
            cluster1, cluster2 = self.simple_split(X, cluster_to_split)

            clusters.append(cluster1)
            clusters.append(cluster2)

        self.assign_labels(clusters, n_samples)


    def calculate_cluster_distance(self, X, cluster1, cluster2):

        distances = []

        for i in cluster1:
            for j in cluster2:
                dist = np.linalg.norm(X[i] - X[j])
                distances.append(dist)

        if self.linkage == 'single':
            return min(distances)

        elif self.linkage == 'complete':
            return max(distances)

        else:
            return sum(distances) / len(distances)

    def cluster_spread(self, X, cluster):

        max_dist = 0

        for i in cluster:
            for j in cluster:
                dist = np.linalg.norm(X[i] - X[j])
                if dist > max_dist:
                    max_dist = dist

        return max_dist

    def simple_split(self, X, cluster):

        max_dist = 0
        point1 = cluster[0]
        point2 = cluster[0]

        for i in cluster:
            for j in cluster:
                dist = np.linalg.norm(X[i] - X[j])
                if dist > max_dist:
                    max_dist = dist
                    point1 = i
                    point2 = j

        cluster1 = []
        cluster2 = []

        for i in cluster:
            dist1 = np.linalg.norm(X[i] - X[point1])
            dist2 = np.linalg.norm(X[i] - X[point2])

            if dist1 < dist2:
                cluster1.append(i)
            else:
                cluster2.append(i)

        return cluster1, cluster2

    def assign_labels(self, clusters, n_samples):

        self.labels = np.zeros(n_samples)

        for cluster_index in range(len(clusters)):
            for sample_index in clusters[cluster_index]:
                self.labels[sample_index] = cluster_index