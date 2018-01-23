import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs


def plot_data(X, y):
    '''
    散点图
    '''
    fig = plt.figure()
    x1 = X[np.where(y == 1)]
    x2 = X[np.where(y == 0)]
    # 添加引用，使得散点图可以画多类点
    # https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot
    ax = fig.add_subplot(111)
    ax.scatter(x1[:, 0], x1[:, 1], s=75, marker='s',
               c='b', alpha=0.5, label='y = 1')
    ax.scatter(x2[:, 0], x2[:, 1], s=75, marker='o',
               c='r', alpha=0.5, label='y = 0')
    plt.legend(loc='upper right')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()


def sigmoid(z)->np.float:
    '''
    自己实现的sigmoid函数
    '''
    h = np.zeros((z.shape))
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def cost_function(theta, X, y, lambda_):
    '''
    带正则项的损失函数
    '''
    m = len(y)
    h = sigmoid(X@theta)
    J = (-y.T@np.log(h) - (1 - y).T@np.log(1 - h)) / m + lambda_ * theta.T@theta / (2 * m)
    return J


def grad_function(theta, X, y, lambda_):
    m = len(y)
    grad = np.zeros(theta.shape[0])
    h = sigmoid(X@theta)
    # 避免第一项也被正则化
    theta[0] = 0
    grad = X.T@(h.T - y).T / m + lambda_ * theta / m
    return grad


def plot_boundary(theta, X, y):
    x1 = X[np.where(y == 1)]
    x1 = x1[:, [1, 2]]
    x2 = X[np.where(y == 0)]
    x2 = x2[:, [1, 2]]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x1[:, 0], x1[:, 1], s=75, marker='s',
               c='b', alpha=0.5, label='y = 1')
    ax.scatter(x2[:, 0], x2[:, 1], s=75, marker='o',
               c='r', alpha=0.5, label='y = 0')

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.shape[0], v.shape[0]))
    for m, i in enumerate(u):
        for n, j in enumerate(v):
            tmp = mapFeature(np.array([i]), np.array([j]))
            z[m, n] = tmp@theta.T
    z = z.T
    ax.contour(u, v, z, [0], linewidth=2)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()


def mapFeature(X1, X2):
    '''
    特征映射，将特征映射到更高的维度使其更好地进行拟合
    此函数为向量化操作，输出为metrix，每行为一个扩展后的特征
    '''
    degree = 6
    out = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.vstack((out, X1 ** (i - j) * X2**j))
    return out.T


def load_data(filename: str, split, dtype):
    '''
    返回txt文件数据
    '''
    return np.loadtxt(fname=filename, delimiter=split, dtype=dtype)


def predict(theta, X):
    '''
    预测函数，将sigmoid结果做映射
    '''
    p = sigmoid(X@theta.T)
    m = X.shape[0]
    for i in range(m):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


if __name__ == '__main__':
    # part1 load the data and plot it
    data = load_data('ex2data2.txt', ',', dtype=np.float64)
    X = data[:, [0, 1]]
    y = data[:, 2]
    plot_data(X, y)

    # part2 regularized logistic regression
    X = mapFeature(X[:, 0], X[:, 1])
    lambda_ = 1
    initial_theta = np.zeros((X.shape[1], 1))
    J = cost_function(initial_theta, X, y, lambda_)
    # 预期的损失应该在0.693左右
    print(J)
    grad = grad_function(initial_theta, X, y, lambda_)
    # 预期的grad前几位为0.0085 0.0188 0.0001 0.0503 0.0115
    print(grad)

    test_theta = np.ones((X.shape[1], 1))
    J = cost_function(test_theta, X, y, 10)
    # 预期损失在3.16左右
    print(J)
    grad = grad_function(test_theta, X, y, 10)
    # 预期的grad前几位为0.3460 0.1614 0.1948 0.2269 0.0922
    print(grad)

    # part3 optimize and plot boundary
    theta = fmin_bfgs(f=cost_function,
                      fprime=grad_function,
                      x0=initial_theta,
                      args=(X, y, lambda_),
                      maxiter=400)
    plot_boundary(theta, X, y)

    # part4 predict
    p = predict(theta, X)
    # 准确率应该在83.1左右
    print(np.mean(p == y) * 100)
