import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.optimize import minimize
from sklearn.svm import SVC


def load_data(filename):
    data = io.loadmat(filename)
    return data


def plot_data(X, y):
    pos = X[np.where(y == 1)]
    neg = X[np.where(y == 0)]
    plt.scatter(pos[:, 0], pos[:, 1], c='k', linewidths=1, marker='+')
    plt.scatter(neg[:, 0], neg[:, 1], c='y', linewidths=1, marker='o')


def gaussian_kernel(X1, X2, sigma=0.1):
    '''
    X1,X2为向量，函数为核函数
    '''
    sub = (X1-X2)**2
    gamma = 2 * (sigma)**2
    return np.exp(-np.sum(sub/gamma))


def kernel_matrix(X1, X2, kernel_func):
    X1 = np.matrix(X1)
    X2 = np.matrix(X2)
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i, m in enumerate(X1):
        for j, n in enumerate(X2):
            K[i, j] = kernel_func(m, n)
    return K


def svm_train(X, y, C, kernel_function, max_iter=-1, tol=1e-3, gamma=1):
    '''
    svm模型训练，可以使用自己定义的kelnel函数，也可以使用sklearn中的核函数
    '''
    if kernel_function == 'gaussianKernel':
        model = SVC(C=C, kernel='precomputed', tol=tol,
                    max_iter=max_iter, gamma=gamma)
        K_x = kernel_matrix(X, X, gaussian_kernel)
        model.fit(K_x)
        return model
    else:
        model = SVC(C=C, kernel=kernel_function, tol=tol,
                    max_iter=max_iter, gamma=gamma)
        model.fit(X, y)
        return model


def visualize_boundary(X, y, model, class_='linear'):
    plot_data(X, y)
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
        plt.contour(X, Y, vals, 0, colors='blue')
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
            model = SVC(C=CTest, tol=1e-3, gamma=gamma).fit(X, y)
            predict = model.predict(Xval)
            curr_error = np.mean((predict != yval).astype('float'))
            if curr_error < errors:
                C = CTest
                sigma = sigmaTest
                errors = curr_error
    return C, sigma


if __name__ == '__main__':
    # part1 load and visualizing data
    data = load_data('./ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)
    # plt.show()

    input('next step')
    plt.close()

    # part2 train linear SVM
    data = load_data('./ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    C = 1
    model = svm_train(X, y, C, 'linear', tol=1e-3, max_iter=20)
    # visualize_boundary(X, y, model, 'linear')

    input('next step')
    plt.close()

    # part3 implement gaussian kernel
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    # 预期值为0.3245
    print(sim)

    # part4 visualizing dataset
    data = load_data('./ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)
    # plt.show()

    input('next step')
    plt.close()

    # part5 train SVM with RBF kernel
    data = io.loadmat('./ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()
    C = 1
    sigma = 0.1
    gamma = 1.0 / (2*(sigma)**2)
    model = svm_train(X, y, C, 'rbf', gamma=gamma)
    # visualize_boundary(X, y, model, class_='gaussian')

    input('next step')
    plt.close()

    # part6 visualizing dataset3
    data = io.loadmat('./ex6data3.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)
    plt.show()

    input('next step')
    plt.close()

    # part7 traom SVM with RBF kernel (dataset3)
    X = data['X']
    y = data['y'].flatten()
    Xval = data['Xval']
    yval = data['yval'].flatten()
    C, sigma = dataset3Params(X, y, Xval, yval)
    gamma = 1.0 / (2*(sigma)**2)
    model = svm_train(X, y, C, 'rbf', gamma=gamma)
    visualize_boundary(X, y, model, class_='gaussian')
