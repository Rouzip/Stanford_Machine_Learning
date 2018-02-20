from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy import io
from scipy.special import expit
from scipy.optimize import minimize


def plot_data(X, example_width=None):
    # 如果图片宽度未指定，则默认使用根号数量作为默认值
    plt.close()
    plt.figure()
    m, n = X.shape
    if not example_width:
        example_width = np.int32(np.round(np.sqrt(n)))
    # 显示灰度图像
    plt.set_cmap('gray')
    example_height = np.int32(n / example_width)

    display_rows = np.int32(np.floor(np.sqrt(m)))
    display_cols = np.int32(np.ceil(m / display_rows))

    pad = 1
    # 展示图片的数组
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            # order = 'F'使其列优先，否则numpy默认行优先
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


def sigmoid(z):
    '''
    scipy中的sigmoid函数
    '''
    return expit(z)


def sigmoid_gradient(z):
    '''
    sigmoid的梯度函数
    '''
    return sigmoid(z) * (1 - sigmoid(z))


def nn_cost_funciton(nn_params,
                     input_layer_size,
                     hidden_layer_size,
                     num_labels,
                     X, y, lambda_, grad_value=False):
    m = X.shape[0]
    # 将系数矩阵还原

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # 计算正向损失
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1@Theta1.T
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    h = sigmoid(a2@Theta2.T)

    # 将y值矩阵化
    all_y = np.zeros((m, num_labels))
    for i in range(m):
        all_y[i, y[i] - 1] = 1
    cost_tmp = - (all_y * np.log(h)) - (1 - all_y) * np.log(1 - h)
    Theta1_nobias = Theta1[:, 1:]
    Theta2_nobias = Theta2[:, 1:]
    reg = np.sum(np.sum(Theta1_nobias ** 2)) + \
        np.sum(np.sum(Theta2_nobias ** 2))
    J = np.sum(np.sum(cost_tmp, axis=1)) / m + lambda_ * reg / (2 * m)

    delta3 = h - all_y
    delta2 = (delta3@Theta2)[:, 1:] * sigmoid_gradient(z2)
    Delta1 = delta2.T@a1
    Delta2 = delta3.T@a2
    Theta1_grad = Delta1 / m + lambda_ * np.hstack((
        np.zeros((hidden_layer_size, 1)), Theta1[:, 1:])) / m
    Theta2_grad = Delta2 / m + lambda_ * np.hstack((
        np.zeros((num_labels, 1)), Theta2[:, 1:])) / m
    grad = np.concatenate((Theta1_grad.reshape((-1, 1)),
                           Theta2_grad.reshape((-1, 1))))
    if grad_value:
        return J, grad
    else:
        return J


def rand_initial_weights(L_in, L_out):
    '''
    随机初始化权重
    '''
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init
    return W


def compute_numberical_gradient(J, theta):
    e = 1e-4
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    for i in range(theta.size):
        perturb[i] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        numgrad[i] = (loss2 - loss1) / (2 * e)
        perturb[i] = 0
    return numgrad


def check_nn_grandients(lambda_=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.array(range(1, m + 1)) % num_labels
    nn_params = np.concatenate((Theta1.reshape(-1, 1),
                                Theta2.reshape(-1, 1)))

    def cost_func(p):
        return nn_cost_funciton(p, input_layer_size, hidden_layer_size,
                                num_labels, X, y, lambda_, True)

    J, grad = cost_func(nn_params)
    numgrad = compute_numberical_gradient(cost_func, nn_params)
    for numerical, analytical in zip(numgrad, grad):
        print((numerical, analytical))

    # 科学计数法显示误差
    diff = Decimal(np.linalg.norm(numgrad - grad)) / \
        Decimal(np.linalg.norm(numgrad + grad))
    print('误差应小于e-11')
    print(diff)


def debug_initialize_weights(fan_out, fan_in):
    # 权重矩阵的形状
    W_shape = ((fan_out, fan_in + 1))
    # 权重矩阵使用sin来作为计算值
    W = np.sin(range(1, fan_out * (fan_in + 1) + 1)
               ).reshape(W_shape, order='F') / 10
    return W


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = sigmoid(a1@Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), z2))
    z3 = sigmoid(a2@Theta2.T)
    index = np.argmax(z3, axis=1)
    return index


if __name__ == '__main__':
    # part1 加载并可视化数据
    data = io.loadmat('./ex4data1.mat')
    X = data['X']
    y = data['y'].flatten()

    # 随机化数据取前100个进行展示
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
    plot_data(sel)
    input('next step')

    # part2 加载系数
    weight = io.loadmat('ex4weights.mat')
    Theta1 = weight['Theta1']
    Theta2 = weight['Theta2']

    # part3 计算正向损失
    nn_params = np.concatenate((Theta1.reshape(-1, 1),
                                Theta2.reshape(-1, 1)))

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    # 不带正则项
    lambda_ = 0
    J = nn_cost_funciton(nn_params,
                         input_layer_size,
                         hidden_layer_size,
                         num_labels,
                         X, y, lambda_)
    print('预期值为0.287629')
    print(J)
    input('next step')

    # part4 实现正则化
    lambda_ = 1
    J = nn_cost_funciton(nn_params,
                         input_layer_size,
                         hidden_layer_size,
                         num_labels,
                         X, y, lambda_)
    print('预期值为0.383770')
    print(J)
    input('next step')

    # part5 sigmoid函数梯度
    # 实验sigmoid_gradient函数
    print('-1, -0.5, 0, 0.5, 1的gradient：')
    print(sigmoid_gradient([-1, -0.5, 0, 0.5, 1]))
    input('next step')

    # part6 初始化参数
    initial_Theta1 = rand_initial_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initial_weights(hidden_layer_size, num_labels)
    initial_nn_params = np.concatenate((initial_Theta1.reshape(-1, 1),
                                        initial_Theta2.reshape(-1, 1)))

    # part7 实现反向传播
    # 生成矩阵用来检查backgrad
    check_nn_grandients()
    input('next step')

    # 添加正则项
    lambda_ = 3
    check_nn_grandients(lambda_)
    print('lambda为3时预期值为0.5756051')

    debug_J = nn_cost_funciton(nn_params, input_layer_size,
                               hidden_layer_size, num_labels, X, y, lambda_)
    print('debug_J:', debug_J)
    input('next step')

    # part8 nn训练
    def costFunc(p):
        return nn_cost_funciton(p, input_layer_size, hidden_layer_size,
                                num_labels, X, y, lambda_, True)

    lambda_ = 1
    options = {'maxiter': 30, 'disp': True}
    myargs = (input_layer_size, hidden_layer_size,
              num_labels, X, y, lambda_, True)
    # 使用L-BFGS-B算法进行优化，得出结果
    result = minimize(nn_cost_funciton, method='L-BFGS-B', args=myargs,
                      x0=initial_nn_params, jac=True, options=options)
    result_nn_params = result['x']
    Theta1_result = np.reshape(
        result_nn_params[:hidden_layer_size * (input_layer_size + 1)],
        (hidden_layer_size, input_layer_size + 1))
    Theta2_result = np.reshape(
        result_nn_params[hidden_layer_size * (input_layer_size + 1):],
        (num_labels, hidden_layer_size + 1))

    # part9 可视化权重
    # 将结果进行展示
    plot_data(Theta1_result[:, 1:])

    # part10 实现预测
    predict_y = predict(Theta1_result, Theta2_result, X)
    # 因为python是以0开始作为索引，所以根据结果特性，预测结果整体+1
    print('准确率为{}%'.format(np.mean(predict_y + 1 == y) * 100))
