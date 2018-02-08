import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import colors


def find_closest_centroids(X, initial_centroids):
    K = initial_centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        distance = np.inf
        for j in range(K):
            tmp = np.sum((X[i, :] - initial_centroids[j, :])**2)
            if tmp < distance:
                distance = tmp
                idx[i] = j
    return idx


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i, :] = np.mean(X[np.where(idx == i), :], axis=1)[
            0].reshape(1, -1)
    return centroids


def run_kmeans(X, initial_centroids, max_iters, plot_progress):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in range(max_iters):
        print('K-Means iteration {}/{}...\n'.format(i, max_iters))
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plt = plot_process_kmeans(X, centroids, previous_centroids)
            previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    if plot_progress:
        plt.show()
    return centroids, idx


def plot_process_kmeans(X, centroids, previous_centroids):
    '''
    画出散点图，并画出中点变化的过程
    '''
    plt.scatter(X[:, 0], X[:, 1])
    plt.plot(previous_centroids[:, 0], previous_centroids[
             :, 1], c='r', marker='x', markersize=10, linewidth=5)
    plt.plot(centroids[:, 0], centroids[:, 1], c='r',
             marker='x', markersize=10, linewidth=5)
    for i in range(centroids.shape[0]):
        p1 = centroids[i, :]
        p2 = previous_centroids[i, :]
        plt.plot(p1[0], p2[0], p1[1], p2[1], '->', linewidth=2)
    return plt


if __name__ == '__main__':
    # part1 find cloest centroids
    data = io.loadmat('./ex7data2.mat')
    K = 3
    X = data['X']
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(X, initial_centroids)
    # print('Closest centroids for the first 3 examples: \n')
    # 预期值为0,2,1
    # print(idx[:3])
    # input('next step')

    # part2 compute means
    centroids = compute_centroids(X, idx, K)
    # 预期值为:2.428301 3.157924  5.813503 2.633656  7.119387 3.616684
    print(centroids)

    # part3 K-means clustering
    data = io.loadmat('./ex7data2.mat')
    K = 3
    max_iter = 10
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    centroids, idx = run_kmeans(X, initial_centroids, max_iter, True)
