import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.optimize import minimize


def load_data(filename):
    data = io.loadmat(filename)
    return data


def plot_data(X, y):
    pos = X[np.where(y == 1)]
    neg = X[np.where(y == 0)]
    plt.scatter(pos[:, 0], pos[:, 1], c='k', linewidths=1, marker='+')
    plt.scatter(neg[:, 0], neg[:, 1], c='y', linewidths=1, marker='o')
    plt.show()


def svm_train(X, y, C, kernel_function,  max_passes=5, tol=1e-3):
    def gaussianKernel(X1, X2, gamma):
        sub = X1.reshape(-1, 1) - X2
        K = np.exp(-sum(sub**2 / (2 * gamma**2)))
        return K

    m, n = X.shape
    y[np.where(y == 0)] = -1
    alphas = np.zeros((m, 1))
    b = 0
    E = np.zeros((m, 1))
    passes = 0
    eta = 0
    L = 0
    H = 0

    if kernel_function == 'linearKernel':
        # 线性核函数
        K = X@X.T
    elif kernel_function == 'gaussianKernel':
        K =


if __name__ == '__main__':
    # part1 load and visualizing data
    data = load_data('./ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)

    input('next step')

    # part2 train linear SVM
    data = load_data('./ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()
    C = 1
