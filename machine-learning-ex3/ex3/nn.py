import numpy as np
import matplotlib.pyplot as plt
from scipy import io

from ex3 import plot_data

if __name__ == '__main__':
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # part1 plot the picture
    data = io.loadmat('./ex3data1.mat')
    X = data['X']
    y = data['y']
    m = X.shape[0]
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
    plot_data(sel)

    # part2 load the weight and predict
