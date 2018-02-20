import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from time import sleep


def load_data(filename):
    data = loadmat(filename)
    return data


def plot_data_orginal(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, y, 'rx', 10, 1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()
    plt.close()


def linear_regcost_function(X, y, theta, lambda_, grad_value=False):
    m = y.shape[0]
    y = y.reshape(-1, 1)
    theta = np.reshape(theta, (-1, y.shape[1]))
    h = X@theta
    reg = lambda_ * (theta[1:]**2).sum() / (2 * m)
    J = 1 / (2 * m) * ((h - y)**2).sum() + reg
    if grad_value:
        tmp_theta = theta.copy()
        tmp_theta[0] = 0
        grad = X.T@(h - y) / m + lambda_ * tmp_theta / m
        return J, grad.flatten()
    else:
        return J


def train_linear_reg(X, y, lambda_):
    def cost_func(p):
        # 简化训练参数输入
        return linear_regcost_function(X, y, p, lambda_, True)
    initital_theta = np.zeros(X.shape[1])
    myoptins = {'maxiter': 200, 'disp': False}
    result = minimize(cost_func, x0=initital_theta, options=myoptins,
                      method='L-BFGS-B', jac=True)
    theta = result['x']
    return theta


def learn_curve(X, y, Xval, yval, lambda_):
    '''
    学习曲线图，将训练集和验证集分别的损失计算并返回
    '''
    m = X.shape[0]
    m_val = Xval.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(m):
        X_train = X[:i + 1]
        y_train = y[:i + 1]
        theta = train_linear_reg(X_train, y_train, lambda_)
        error_train[i] = linear_regcost_function(
            X_train, y_train, theta, lambda_)
        error_val[i] = linear_regcost_function(Xval, yval, theta, lambda_)
    return error_train, error_val[:m]


def ploy_feature(X, p):
    '''
    将X进行多项式展开
    '''
    m = X.shape[0]
    X_poly = np.matrix(np.zeros((m, p)))

    for i in range(p):
        X_poly[:, i] = X**(i + 1)
    return np.array(X_poly)


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    # python默认使用n，而matlab则默认使用n-1
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def plotfit(min_x, max_x, mu, sigma, theta, p, plt):
    X = np.arange(min_x - 15, max_x + 25, 0.05)
    X_poly = ploy_feature(X.reshape(-1, 1), p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    X_poly = np.hstack((np.ones((X.shape[0], 1)), X_poly))
    plt.plot(X, X_poly@theta, '--', linewidth=2)


def validation_curve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    length = len(lambda_vec)
    error_train = np.zeros((length, 1))
    error_val = np.zeros((length, 1))
    for i in range(length):
        lambda_ = lambda_vec[i]
        theta = train_linear_reg(X, y, lambda_)
        error_train[i] = linear_regcost_function(X, y, theta, 0)
        error_val[i] = linear_regcost_function(Xval, yval, theta, 0)
    return lambda_vec, error_train, error_val


if __name__ == '__main__':
    # part0 加载并可视化数据
    data = load_data('./ex5data1.mat')
    X = data['X']
    y = data['y']
    m = X.shape[0]
    train = np.concatenate((np.ones((m, 1)), X), axis=1)
    plot_data_orginal(X, y)
    input('next step')

    # part1 正则化线性回归损失
    theta = np.array([1, 1]).reshape(-1, 1)
    print('预期值为303.993192')
    J = linear_regcost_function(train, y, theta, 1)
    print(J)
    input('next step')

    # part2 正则化线性回归梯度
    print('预期梯度值为-15.303016; 598.250744')
    J, grad = linear_regcost_function(train, y, theta, 1, True)
    print(grad)
    input('next step')

    # part3 训练线性回归
    lambda_ = 0
    theta = train_linear_reg(train, y, lambda_)
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.hstack((np.ones((m, 1)), X))@theta, '--', linewidth=2)
    plt.show()
    input('next step')

    # part4 线性回归学习曲线
    Xval = data['Xval']
    Xval_train = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
    yval = data['yval'].flatten()
    error_train, error_val = learn_curve(train, y,
                                         Xval_train, yval, lambda_)
    xaxis = np.array(range(m))
    plt.close()
    plt.plot(xaxis, error_train, xaxis, error_val)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()

    # part5 多项式回归的特征映射
    p = 8
    X_poly = ploy_feature(X, p)
    X_poly, mu, sigma = feature_normalize(X_poly)
    X_poly = np.hstack((np.ones((m, 1)), X_poly))

    Xtest = data['Xtest']
    ytest = data['ytest']
    X_poly_test = ploy_feature(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

    X_poly_val = ploy_feature(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))
    print('归一化后第一行为：{}'.format(X_poly[0, :]))
    input('next step')

    # part6 多项式回归的学习曲线
    lambda_ = 0
    theta = train_linear_reg(X_poly, y, lambda_)
    print('theta is: {}'.format(theta))
    fig1 = plt.figure(1)
    plt.close()
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plotfit(min(X), max(X), mu, sigma, theta, p, plt)
    plt.show()

    fig2 = plt.figure(2)
    error_train, error_val = learn_curve(X_poly, y, X_poly_val, yval, lambda_)
    plt.close()
    plt.plot(xaxis, error_train, xaxis, error_val)
    plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(lambda_))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(['Train', 'Cross Validation'])
    plt.axis([0, 13, 0, 100])
    plt.show()

    print('Polynomial Regression (lambda = {})\n\n'.format(lambda_))
    print('#Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('\t {}\t\t{}\t{}\n'.format(i + 1, error_train[i], error_val[i]))

    input('next step')

    # part7 对于lambda进行挑选
    lambda_vec, error_train, error_val = validation_curve(
        X_poly, y, X_poly_val, yval)
    plt.close('all')
    p1, p2 = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()

    print('lambda\t\tTrain Error\tValidation Error\n')
    for i in range(len(lambda_vec)):
        print(' {}\t{}\t{}\n'.format(
            lambda_vec[i], error_train[i], error_val[i]))
    input('next step')

    # part8 将最优lambda在测试集中计算损失
    lambda_val = 3
    theta = train_linear_reg(X_poly, y, lambda_val)
    error_test = linear_regcost_function(X_poly_test, ytest, theta, 0)
    # 预期值为3.859
    print('Test set error :{}\n'.format(error_test))
    input('next step')

    # part9 随机挑选几个例子进行学习曲线可视化
    lambda_val = 0.01
    times = 50

    error_train_rand = np.zeros((m, times))
    error_val_rand = np.zeros((m, times))

    for i in range(1, m + 1):
        for j in range(times):
            # 随机从X_ploy和X_val中选取一些索引训练theta
            rand_sample = np.random.permutation(m)
            rand_train_index = rand_sample[:i]
            X_train_rand = X_poly[rand_train_index, :]
            y_train_rand = y[rand_train_index]

            rand_sample = np.random.permutation(X_poly_test.shape[0])
            rand_val_index = rand_sample[:i]
            X_val_rand = X_poly_val[rand_val_index, :]
            y_val_rand = yval[rand_val_index]

            theta = train_linear_reg(X_train_rand, y_train_rand, lambda_val)
            error_train_rand[i - 1, j] = \
                linear_regcost_function(X_train_rand, y_train_rand, theta, 0)
            error_val_rand[i - 1, j] = \
                linear_regcost_function(X_val_rand, y_val_rand, theta, 0)
    error_train = np.mean(error_train_rand, axis=1)
    error_val = np.mean(error_val_rand, axis=1)

    # 可视化曲线
    plt.close('all')
    p1, p2 = plt.plot(xaxis, error_train, xaxis, error_val)
    plt.title(
        'Polynomial Regression Learning Curve (lambda = {})'.format(lambda_val))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend((p1, p2), ('Train', 'Cross Validation'))
    plt.axis([0, 13, 0, 150])
    plt.show()

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('  \t{:d}\t\t{:f}\t{:f}\n'.format(
            i + 1, error_train[i], error_val[i]))
