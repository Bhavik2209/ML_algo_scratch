import numpy as np

class DBSCAN:

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):

        n_samples = X.shape[0]

        self.labels = [-1] * n_samples
        visited = [False] * n_samples

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self.get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self.expand_clsuter(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def expand_clsuter(self, X, index, neighbors, cluster_id, visited):
        self.labels[index] = cluster_id

        i=0
        while i < len(neighbors):
            point = neighbors[i]

            if not visited[point]:
                visited[point] = True
                point_neighbors = self.get_neighbors(X, point)

                if len(point_neighbors) >= self.min_samples:
                    neighbors += point_neighbors

            if self.labels[point] == -1:    
                self.labels[point] = cluster_id # border point
            
            i += 1



    def get_neighbors(self, X, index):
        neighbors = []

        for j in range(X.shape[0]):
            distance = np.linalg.norm(X[index] - X[j])

            if distance <= self.eps:
                neighbors.append(j)
        
        return neighbors
    


import numpy as np

# two small clusters + one noise point
X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [8, 9],
    [9, 8],
    [20, 20]   # noise
])

db = DBSCAN(eps=1.5, min_samples=2)
db.fit(X)

print(db.labels)