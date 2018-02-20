import random
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

from ex3 import plot_data, sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = sigmoid(a1@Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), z2))
    z3 = sigmoid(a2@Theta2.T)
    index = np.argmax(z3, axis=1)
    return index


if __name__ == '__main__':
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # part1 加载并可视化数据
    data = io.loadmat('./ex3data1.mat')
    X = data['X']
    y = data['y'].flatten()
    m = X.shape[0]
    # 打乱从中选取前100个数字
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
    plot_data(sel)
    input('next step')

    # part2 加载系数
    weight = io.loadmat('ex3weights.mat')
    Theta1 = weight['Theta1']
    Theta2 = weight['Theta2']

    # part3 实现预测
    pred = predict(Theta1, Theta2, X)
    result = np.mean((pred + 1) % 10 == y % 10) * 100
    print('准确率为：' + str(result))
    input('next step')

    # 初始化时间种子
    random.seed(time.time())
    # 从5000个图片之中选择一幅
    num = random.choice(range(5000))
    plot_data(X[num, :].reshape((1, -1)))
    pred = predict(Theta1, Theta2, X[num, :].reshape((1, -1)))
    # 下标的原因，所以使用+1再%10
    print('数字为 ' + str((pred + 1) % 10))
