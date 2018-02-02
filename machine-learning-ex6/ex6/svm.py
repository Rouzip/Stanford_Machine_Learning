from sklearn import svm
from scipy import io
import numpy as np
from matplotlib import pyplot as plt


def SVM():
    '''线性分类'''
    data1 = io.loadmat('ex6data1.mat')
    X = data1['X']
    y = data1['y'].ravel()
    plt = plot_data(X, y)

    model = svm.SVC(kernel='linear').fit(X, y)
    plot_decision_boundary(X, y, model)

    '''非线性分类'''
    data2 = io.loadmat('ex6data2.mat')
    X = data2['X']
    y = data2['y'].ravel()
    plt = plot_data(X, y)

    model = svm.SVC(gamma=100).fit(X, y)
    plot_decision_boundary(X, y, model, class_='notLinear')

    '''
    最后一个实验
    '''
    data3 = io.loadmat('ex6data3.mat')
    X = data3['X']
    y = data3['y'].ravel()
    Xval = data3['Xval']
    yval = data3['yval'].ravel()
    plt = plot_data(X, y)
    C, sigma = dataset3Params(X, y, Xval, yval)
    gamma = 1. / (2. * sigma ** 2)
    model = svm.SVC(C=C, gamma=gamma).fit(X, y)
    print(model)
    plot_decision_boundary(X, y, model, class_='notLinear')


def plot_data(X, y):
    '''
    画图
    '''
    plt.figure(figsize=(10, 8))
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1, = plt.plot(np.ravel(X[pos, 0]), np.ravel(
        X[pos, 1]), 'ro', markersize=8)
    p2, = plt.plot(np.ravel(X[neg, 0]), np.ravel(
        X[neg, 1]), 'g^', markersize=8)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend([p1, p2], ["y==1", "y==0"])
    return plt


def plot_decision_boundary(X, y, model, class_='linear'):
    plt = plot_data(X, y)
    if class_ == 'linear':
        # 线性画边缘
        w = model.coef_
        b = model.intercept_
        xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        yp = -(w[0, 0] * xp + b) / w[0, 1]
        plt.plot(xp, yp, 'b-', linewidth=2.0)
        plt.show()
    else:
        # 画等高线边缘
        x_1 = np.transpose(np.linspace(
            np.min(X[:, 0]), np.max(X[:, 0]), 100).reshape(1, -1))
        x_2 = np.transpose(np.linspace(
            np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(1, -1))
        # 网格
        X, Y = np.meshgrid(x_1, x_2)
        vals = np.zeros(X.shape)
        for i in range(X.shape[1]):
            this_X = np.hstack(
                (X[:, i].reshape(-1, 1), Y[:, i].reshape(-1, 1)))
            vals[:, i] = model.predict(this_X)
        plt.contour(X, Y, vals, [0, 1], colors='blue')
        plt.show()


def dataset3Params(X, y, Xval, yval):
    '''
    遍历求出最好的系数
    '''
    choices = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    errors = 100000.
    for CTest in choices:
        for sigmaTest in choices:
            gamma = 1. / (2. * sigmaTest ** 2)
            model = svm.SVC(C=CTest, tol=1e-3, gamma=gamma).fit(X, y)
            predict = model.predict(Xval)
            curr_error = np.mean((predict != yval).astype('float'))
            if curr_error < errors:
                C = CTest
                sigma = sigmaTest
                errors = curr_error
    return C, sigma


if __name__ == '__main__':
    SVM()
