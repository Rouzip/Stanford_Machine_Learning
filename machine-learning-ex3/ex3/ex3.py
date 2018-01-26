import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.optimize import fmin_bfgs
from scipy import optimize


def plot_data(X, example_width=None):
    # 假如图片宽度未指定，使用矩阵行向量开根号的长度
    plt.close()
    plt.figure()
    m, n = X.shape
    if not example_width:
        example_width = np.int32(np.round(np.sqrt(n)))
    # 展示灰度图像
    plt.set_cmap('gray')
    example_height = np.int32(n / example_width)

    # 每行展示多少和每列展示多少图片
    display_rows = np.int32(np.floor(np.sqrt(m)))
    display_cols = np.int32(np.ceil(m / display_rows))

    pad = 1
    # 初始化空数组用于后面结合展示图片
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            # order = 'F'是为了保证图像是正的，否则是转置的
            display_array[pad + i * (example_height + pad):
                          pad + i * (example_height + pad) + example_height,
                          pad + j * (example_width + pad):
                          pad + j * (example_width + pad) + example_width] =\
                X[curr_ex, :].reshape(
                    example_height, example_width, order='F') / max_val
            curr_ex += 1
        if curr_ex >= m:
            break
    plt.axis('off')
    plt.imshow(display_array)
    plt.show()


def sigmoid(z)->np.float:
    '''
    自己实现的sigmoid函数
    '''
    h = np.zeros((z.shape))
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def lr_cost_function(theta, X, y, lambda_, return_grad=False):
    m = y.shape[0]
    h = X@theta
    theta_t = theta.copy()
    theta_t[0] = 0
    regularized = theta_t@theta_t.T * float(lambda_) / (2 * m)
    J = (-(y@np.log(sigmoid(h)).T) -
         (1 - y)@np.log(1 - sigmoid(h)).T) / m + regularized
    regularized = float(lambda_) / m * theta_t
    grad = X.T@(sigmoid(h) - y).T / m + regularized
    if return_grad:
        return J, grad.flatten()
    else:
        return J


def grad_function(theta, X, y, lambda_):
    m = y.shape[0]
    h = X@theta.T
    theta_t = theta.copy()
    theta_t[0] = 0
    regularized = float(lambda_) / m * theta_t
    grad = (sigmoid(h).T - y).T@X / m + regularized
    return grad


def one_vs_all(X, y, num_labels, lambda_):
    m, n = X.shape
    train = np.column_stack((np.ones((m, 1)), X))
    all_theta = np.zeros((num_labels, n + 1))
    for i in range(num_labels):
        initial_theta = np.zeros((n + 1, 1))
        # 训练的数据集和正则化参数，由于选用了minimize，而且jac为true，所以函数中需要有梯度
        myargs = (train, np.int32(y % 10 == i), lambda_, True)
        theta = optimize.minimize(lr_cost_function,
                                  x0=initial_theta,
                                  args=myargs,
                                  options={'maxiter': 50, 'disp': True},
                                  method='Newton-CG',
                                  jac=True)
        all_theta[i, :] = theta['x']
    return all_theta


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    train = np.hstack((np.ones((m, 1)), X))
    h = sigmoid(train@all_theta.T)
    p = np.argmax(h, axis=1)
    # 选取其中可能性最大的那个作为最后预测的标准
    return p


if __name__ == '__main__':
    # part1 plot the pictures
    # 输入层为展开的20*20图片，分类种类为10种
    input_layer_size = 400
    num_labels = 10
    data = io.loadmat('ex3data1.mat')
    # 5000*400
    X = data['X']
    # 5000
    y = data['y'].flatten()
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
    plot_data(sel)

    # part2 vectorize logistic regression
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.hstack((np.ones((5, 1)), np.array(
        range(1, 16)).reshape((5, 3), order='F') / 10))
    y_t = np.array([True, False, True, False, True])
    lambda_t = 3
    J = lr_cost_function(theta_t, X_t, y_t, lambda_t)
    # 预期值为2.534819
    print(J)
    grad = grad_function(theta_t, X_t, y_t, lambda_t)
    # 预期值为0.146561 -0.548558 0.724722 1.398003
    print(grad)

    # part3 one vs all
    lambda_ = 0.1
    all_theta = one_vs_all(X, y, num_labels, lambda_)
    result = predict_one_vs_all(all_theta, X)
    print(np.mean(result == y % 10))
