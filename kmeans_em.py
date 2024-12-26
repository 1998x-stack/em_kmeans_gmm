
import numpy as np
from typing import Optional

class KMeans:
    def __init__(self, n_clusters: int, tol: float = 1e-6, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None

    def _initialize_centroids(self, X: np.ndarray):
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def fit(self, X: np.ndarray):
        self._initialize_centroids(X)
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            labels = self._assign_clusters(X)
            self.centroids = self._compute_centroids(X, labels)
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)

if __name__ == "__main__":
    data = np.load("gaussian_mixture_data.npy")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    print("K-Means聚类完成。")
