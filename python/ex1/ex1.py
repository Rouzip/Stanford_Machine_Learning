import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.axes3d import Axes3D


def load_data(filename):
    '''
    读取数据，将其转换为np.array的形式，将x和y以元组形式返回，解包获取数据
    '''
    column1 = list()
    column2 = list()
    with open(filename) as fp:
        for line in fp.readlines():
            line = line.strip()
            a, b = line.split(',')
            column1.append(a)
            column2.append(b)
    column1 = np.array(column1).astype(float)
    column2 = np.array(column2).astype(float)
    return column1, column2


def plot_data(X, y, Xlabel, ylabel):
    '''
    将图像进行描绘，散点图
    '''
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.show()


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        t = alpha * np.sum(((X@theta).T - y).T * X, axis=0) / m
        theta = (theta.T - t).T
        J_history[i] = compute_cost(X, y, theta)
    return theta


def compute_cost(X: np.matrix, y: np.matrix, theta: np.matrix)->float:
    m = len(y)
    # 最小二乘法计算损失
    tmp = (X@theta).T - y
    loss = 1 / (2 * m) * tmp@tmp.T
    return loss


if __name__ == '__main__':
    # part1 可视化数据
    # 从文件之中读取数据
    X, y = load_data('./ex1data1.txt')
    plot_data(X, y, 'Profit in $10,000s', 'Population of City in 10,000s')
    input('next step')

    # part2 损失函数和梯度
    # 对数据进行处理，形式变为列的形式
    train = X.T
    # 添加偏置
    train = np.concatenate((np.ones((X.shape[0], 1)), X[:, None]), axis=1)
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    J = compute_cost(train, y, theta)
    print('预期损失32.07')
    print(J)
    J = compute_cost(train, y, np.array([-1, 2]).T)
    print('预期损失54.24')
    print(J)
    theta = gradient_descent(train, y, theta, alpha, iterations)
    print('预期theta为-3.6303  1.1664')
    print('theta', theta.flatten())
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(X, y, s=75, marker='x',
               c='b', alpha=0.5, label='Admitted')
    ax.plot(X, train@theta, '-')
    plt.legend(['Linear regression', 'Training data'])
    plt.show()
    predict1 = np.array([1, 3.5])@theta
    predict2 = np.array([1, 7])@theta
    print('对于35,000 和 70,000人口，做出预测如下: ')
    print(predict1, predict2)
    input('next step')

    # part3 可视化J
    theta1 = np.linspace(-10, 10, 100)
    theta2 = np.linspace(-1, 4, 100)
    # 存储损失，后面画出等线图
    J_all = np.zeros((len(theta1), len(theta2)))
    for i, m in enumerate(theta1):
        for j, n in enumerate(theta2):
            theta = np.array([m, n]).T
            J_all[i][j] = compute_cost(train, y, theta)
    # 将x轴和y轴进行转化
    T1, T2 = np.meshgrid(theta1, theta2)

    pic = plt.figure(2)
    ax = pic.gca(projection='3d')
    # 需要将J_all进行转置才为正确的图
    surf = ax.plot_surface(T1, T2, J_all.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.view_init(elev=15, azim=-118)
    plt.show()
    input('next step')
    plt.close()
    plt.figure(3)
    cs = plt.contour(T1, T2, J_all.T,
                     np.logspace(-2, 3, 20),
                     colors=('r', 'g', 'b', (1, 1, 0), '#afeeee', '0.5'))
    plt.show()
