import numpy as np

class ShilhouetteScore:

    def calculate(X, labels):
        n_samples = X.shape[0]
        unique_clusters = np.unique(labels)

        silhouette_values = []

        for i in range(n_samples):

            x_i = X[i]
            cluster_i = labels[i]

            same_cluster_points = X[labels == cluster_i]

            if len(same_cluster_points) > 1:
                distances = [np.linalg.norm(x_i - x) 
                            for x in same_cluster_points if not np.array_equal(x, x_i)]
                a_i = np.mean(distances)
            else:
                a_i = 0


            b_i = float('inf')

            for cluster in unique_clusters:
                if cluster == cluster_i:
                    continue

                other_cluster_points = X[labels == cluster]
                distances = [np.linalg.norm(x_i - x) 
                            for x in other_cluster_points]
                mean_distance = np.mean(distances)

                b_i = min(b_i, mean_distance)


            if max(a_i, b_i) == 0:
                s_i = 0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)

            silhouette_values.append(s_i)

        return np.mean(silhouette_values)