
import numpy as np
from typing import Optional

class GaussianMixtureModel:
    def __init__(self, n_components: int, tol: float = 1e-6, max_iter: int = 100):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.means = None
        self.covariances = None
        self.weights = None

    def _initialize_parameters(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        likelihoods = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )
        responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        N_k = responsibilities.sum(axis=0)
        self.weights = N_k / X.shape[0]
        self.means = (responsibilities.T @ X) / N_k[:, None]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (
                responsibilities[:, k][:, None] * diff
            ).T @ diff / N_k[k]

    def _multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n = mean.size
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1))
        return exp_term / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))

    def fit(self, X: np.ndarray):
        self._initialize_parameters(X)
        for _ in range(self.max_iter):
            old_means = self.means.copy()
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            if np.linalg.norm(self.means - old_means) < self.tol:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

if __name__ == "__main__":
    data = np.load("gaussian_mixture_data.npy")
    gmm = GaussianMixtureModel(n_components=3)
    gmm.fit(data)
    labels = gmm.predict(data)
    print("GMM聚类完成。")
