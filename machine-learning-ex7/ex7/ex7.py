import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2


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


def run_Kmeans(X, initial_centroids, max_iters, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    for i in range(max_iters):
        print('K-Means iteration {}/{}...\n'.format(i + 1, max_iters))
        # 计算每个点属于哪个类 idx:[m,1]
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plt = plot_process_Kmeans(X, centroids, previous_centroids, idx, K)
            previous_centroids = centroids
            plt.show()
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


def plot_process_Kmeans(X, centroids, previous_centroids, idx, K):
    '''
    画出散点图，并画出中点变化的过程
    '''
    plot_data_points(X, idx, K)
    plt.plot(previous_centroids[:, 0], previous_centroids[
             :, 1], 'rx', markersize=10, linewidth=5)
    plt.plot(centroids[:, 0], centroids[:, 1],
             'rx', markersize=10, linewidth=5)
    for i in range(centroids.shape[0]):
        p1 = centroids[i, :]
        p2 = previous_centroids[i, :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '->', linewidth=2)
    return plt


def Kmeans_randinitial_centroids(X, K):
    randidx = np.random.permutation(np.array(range(K)))
    result = X[randidx, :]
    return result


def plot_data_points(X, idx, K):
    '''
    显示出调色出的散点图
    '''
    # inspired by http://stackoverflow.com/q/23945764/583834
    palette = colors.hsv_to_rgb(
        np.column_stack((np.linspace(0, 1, K + 1).reshape(K + 1, -1),
                         np.ones((K + 1, 2)))))
    colors_map = np.array([palette[int(i)] for i in idx])
    plt.scatter(X[:, 0], X[:, 1], s=75, edgecolors=colors_map)


if __name__ == '__main__':
    # part1 find cloest centroids
    data = io.loadmat('./ex7data2.mat')
    K = 3
    X = data['X']
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(X, initial_centroids)
    print('Closest centroids for the first 3 examples: \n')
    # 预期值为0,2,1
    print(idx[:3])
    input('next step')

    # part2 compute means
    centroids = compute_centroids(X, idx, K)
    # 预期值为:2.428301 3.157924  5.813503 2.633656  7.119387 3.616684
    print(centroids)

    # part3 K-means clustering
    data = io.loadmat('./ex7data2.mat')
    K = 3
    max_iter = 10
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    centroids, idx = run_Kmeans(X, initial_centroids, max_iter, True)

    # part4 K-Means clustering on pixels
    img = cv2.imread('./bird_small.png')
    img = img / 255
    X = img.reshape(img.shape[0] * img.shape[1], 3)
    K = 16
    max_iter = 10
    initial_centroids = Kmeans_randinitial_centroids(X, K)
    centroids, _ = run_Kmeans(X, initial_centroids, max_iter)
    input('next step')

    # part5 Image Compression
    idx = find_closest_centroids(X, centroids)
    idx = np.array([int(i) for i in idx])
    idx = idx.flatten()
    X_recovered = centroids[idx, :]
    X_recovered = X_recovered.reshape(img.shape[0], img.shape[1], 3)
    # 交换通道，cv2和matplotlib默认显示通道不同
    # cv2: r,g,b    matplotlib:b,g,r
    r, g, b = cv2.split(X_recovered)
    img_recover = cv2.merge([b, g, r])
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_recover)
    plt.show()
