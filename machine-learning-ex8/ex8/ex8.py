import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import scipy.linalg as linalg


def estimate_gaussian(X):
    m, _ = X.shape
    mu = np.mean(X, axis=0)

    # var 为方差，cov为协方差
    # 和之前相同，ddof决定Delta Degrees of Freedom
    sigma2 = np.var(X, axis=0, ddof=0)
    return mu, sigma2


def multivariate_gaussian(X, mu, sigma2):
    k = len(mu)
    sigma2 = sigma2.flatten()
    sigma2 = linalg.diagsvd(sigma2, len(sigma2), len(sigma2))

    # 均值化
    X = X - mu

    p = ((2.0 * np.pi)**(-k / 2.0)) * (np.linalg.det(sigma2)**(-0.5)) *\
        np.exp(-0.5 * np.sum(X@np.linalg.pinv(sigma2) * X, axis=1))

    return p


def visualizefit(X, mu, sigma2):
    axis_range = np.linspace(0, 35, num=71)
    x1, x2 = np.meshgrid(axis_range, axis_range)
    Z = multivariate_gaussian(
        np.hstack((x1.reshape(-1, 1, order='F'),
                   x2.reshape(-1, 1, order='F'))),
        mu, sigma2)
    Z = Z.reshape(x1.shape, order='F')
    plt.plot(X[:, 0], X[:, 1], 'bx')
    np.arange(-20, 0, 3)
    if np.sum(np.isinf(Z)) == 0:
        # numpy不允许直接将int作为底数，所以需要指定10.0
        plt.contour(x1, x2, Z, np.power(10.0, np.arange(-20, 0, 3)))


def select_threshold(yval, pval):
    best_F1 = 0
    step_size = (np.max(pval) - np.min(pval)) / 1000
    steps = np.arange(np.min(pval), np.max(pval), step_size)
    for epsilon in steps:
        cvPrediction = pval < epsilon

        # 根据定义计算出tp,fp,fn
        tp = np.sum((cvPrediction == True) * (yval == True))
        fp = np.sum((cvPrediction == True) * (yval == False))
        fn = np.sum((cvPrediction == False) * (yval == True))

        prec = tp / (tp + fp) if (tp + fp) != 0 else np.inf
        rec = tp / (tp + fn) if (tp + fn) != 0 else np.inf
        # RuntimeWarning: invalid value encountered in double_scalars
        # 无法解决，由于被除数为inf导致
        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_F1, best_epsilon


if __name__ == '__main__':
    # part1 load example dataset
    data = io.loadmat('./ex8data1.mat')
    X = data['X']
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.axis([0, 30, 0, 30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
    input('next step')
    plt.close()

    # part2 extimate the dataset statistics
    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    visualizefit(X, mu, sigma2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
    input('next step')
    plt.close()

    # part3 find outliers
    Xval = data['Xval']
    yval = data['yval'].flatten()
    pval = multivariate_gaussian(Xval, mu, sigma2)
    F1, epsilon = select_threshold(yval, pval)
    outliers = np.where(p < epsilon)
    visualizefit(X, mu, sigma2)
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
    plt.show()
    input('next step')
    plt.close()

    # part4 mutidimensional outliers
    data = io.loadmat('./ex8data2.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval'].flatten()
    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    pval = multivariate_gaussian(Xval, mu, sigma2)
    F1, epsilon = select_threshold(yval, pval)

    # 预期值为0.615385
    print(F1)
    # 预期值为1.38e-18
    print(epsilon)
    # 预期值为117
    print(np.sum(p < epsilon))
