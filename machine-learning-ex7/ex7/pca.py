import numpy as np
import matplotlib.pyplot as plt
from scipy import io


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def pac(X):
    m = X.shape[0]
    cov = X.T@X / m
    U, V, _ = np.linalg.svd(cov)
    return U, V


if __name__ == '__main__':
    # part1 load example dataset
    data = io.loadmat('./ex7data1.mat')
    X = data['X']
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='o')
    # plt.show()
    # input('next step')

    # part2 principal componment analysis
    X_norm, mu, sigma = feature_normalize(X)
    U, S = pac(X_norm)
    print(U[0, 0], U[1, 0])
