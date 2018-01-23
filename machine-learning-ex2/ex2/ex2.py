import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs


def load_data(filename: str, split, dtype):
    '''
    返回txt文件数据
    '''
    return np.loadtxt(fname=filename, delimiter=split, dtype=dtype)


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
               c='b', alpha=0.5, label='Admitted')
    ax.scatter(x2[:, 0], x2[:, 1], s=75, marker='o',
               c='r', alpha=0.5, label='Not admitted')
    plt.legend(loc='upper right')
    plt.show()


def sigmoid(z)->np.float:
    '''
    自己实现的sigmoid函数
    '''
    h = np.zeros((z.shape))
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def cost_function(theta, X, y):
    '''
    计算损失函数
    '''
    m = len(y)
    z = np.dot(X, theta)
    J = 1.0 / m * (-np.dot(y.T, np.log(sigmoid(z))) -
                   np.dot((1 - y).T, np.log(1 - sigmoid(z))))
    return J


def grad_function(theta, X, y):
    '''
    计算梯度
    '''
    m = len(y)
    z = np.dot(X, theta)
    grad = np.zeros((theta.shape[0], 1))
    grad = 1.0 / m * np.dot(X.T, sigmoid(z) - y.T)
    return grad


def plot_boundary(theta, X, y):
    '''
    画出决策边界
    '''
    # 去掉偏置项
    X = X[:, 1:]
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    x1 = X[np.where(y == 1)]
    x2 = X[np.where(y == 0)]
    ax.scatter(x1[:, 0], x1[:, 1], s=75, marker='s',
               c='b', alpha=0.5, label='Admitted')
    ax.scatter(x2[:, 0], x2[:, 1], s=75, marker='s',
               c='r', alpha=0.5, label='Admitted')

    # 两个端点确定直线
    print('size X', X.shape)
    plot_x = [np.min(X[:, 1], axis=0) - 2, np.max(X[:, 1], axis=0)]
    plot_y = np.array(
        (-1.0 / theta[2]) * (np.dot(theta[1], plot_x) + theta[0]))
    print(plot_x, plot_y)
    ax.plot(plot_x, plot_y)
    plt.axis([30, 100, 30, 100])

    plt.show()


def predict(theta, X):
    '''
    预测函数，将sigmoid结果做映射
    '''
    p = sigmoid(np.dot(X, theta.T))
    m = X.shape[0]
    for i in range(m):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


if __name__ == '__main__':
    # part1 plot origin data
    data = load_data('./ex2data1.txt', ',', np.float64)
    X = np.array(data[:, 0:-1], dtype=np.float64)
    y = np.array(data[:, -1], dtype=np.float64)
    plot_data(X, y)

    # part2 sigmoid regression
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    initial_theta = np.zeros((X.shape[1], 1))
    J = cost_function(initial_theta, X, y)
    # 正确的损失值为0.693
    print(J)
    test_theta = np.array([-24, 0.2, 0.2]).T
    J = cost_function(test_theta, X, y)
    # 正确的损失值为0.218
    print(J)

    # part3 optimzing the theta
    theta = fmin_bfgs(f=cost_function,
                      fprime=grad_function,
                      x0=np.zeros((3, 1)),
                      args=(X, y),
                      maxiter=400)
    # 画出决策边界
    plot_boundary(theta, X, y)

    # part3 predict and accuration
    # 测试的45 85分，估计值应该在0.775 +/- 0.002
    tmp_x = np.array([1, 45, 85])
    print(sigmoid(np.dot(tmp_x, theta)))

    # 预测准确率,
    p = predict(theta, X)
    print(np.mean(p == y) * 100)
