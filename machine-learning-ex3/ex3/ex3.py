import numpy as np
import matplotlib.pyplot as plt
from scipy import io


if __name__ == '__main__':
    # 输入层为展开的20*20图片，分类种类为10种
    input_layer_size = 400
    num_labels = 10
    data = io.loadmat('ex3data1.mat')
    # 5000*400
    X = data['X']
    # 5000*1
    y = data['y']
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
