
import numpy as np
from sklearn.datasets import make_spd_matrix

def generate_gaussian_mixture_data(
    means: list[np.ndarray],
    covariances: list[np.ndarray],
    weights: list[float],
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)
    data = np.vstack([
        np.random.multivariate_normal(mean, cov, int(n_samples * weight))
        for mean, cov, weight in zip(means, covariances, weights)
    ])
    return data

if __name__ == "__main__":
    means = [np.array([0, 0]), np.array([5, 5]), np.array([10, 0])]
    covariances = [np.eye(2), make_spd_matrix(2), make_spd_matrix(2)]
    weights = [0.3, 0.4, 0.3]
    data = generate_gaussian_mixture_data(means, covariances, weights, n_samples=10000)
    np.save("gaussian_mixture_data.npy", data)
    print("数据已成功生成并保存到 gaussian_mixture_data.npy 文件中。")
