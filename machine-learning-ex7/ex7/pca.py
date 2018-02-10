import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.linalg import diagsvd
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import cv2


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def pca(X):
    m = X.shape[0]
    cov = X.T@X / m
    U, S, _ = np.linalg.svd(cov)
    # 将特征值转换为矩阵形式
    # https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.linalg.diagsvd.html#scipy.linalg.diagsvd
    S = diagsvd(S, len(S), len(S))
    return np.matrix(U), np.matrix(S)


def drawline(p1, p2, *args, **kargs):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-k', linewidth=2)


def project_data(X, U, K):
    U_reduce = U[:, :K]
    Z = X@U_reduce
    return Z


def recover_data(Z, U, K):
    X_rec = Z@U[:, :K].T
    return X_rec


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
        np.column_stack([np.linspace(0, 1, K + 1).reshape(K + 1, -1, order='F'),
                         np.ones((K + 1, 2))]))
    colors_map = np.array([palette[int(i)] for i in idx])
    a = [X[:, 0]]
    b = [X[:, 1]]
    plt.scatter(a, b, s=75,  facecolors='none', edgecolors=colors_map)


if __name__ == '__main__':
    # part1 load example dataset
    data = io.loadmat('./ex7data1.mat')
    X = data['X']
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='o')
    plt.axis([0.5, 6.5, 2, 8])
    plt.show()
    input('next step')

    # part2 principal componment analysis
    X_norm, mu, sigma = feature_normalize(X)
    U, S = pca(X_norm)
    # 预期值为-0.707107 -0.707107
    print(U[0, 0], U[1, 0])

    plt.close()
    plt.scatter(X[:, 0], X[:, 1], c='b', marker='o')
    p1 = mu
    p2 = np.array(mu + 1.5 * S[0, 0] * U[:, 0].T).flatten()
    drawline(p1, p2)
    p2 = np.array(mu + 1.5 * S[1, 1] * U[:, 1].T).flatten()
    drawline(p1, p2)
    plt.show()

    # part3 dimension reduction
    plt.close()
    plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
    plt.axis([-4, 3, -4, 3])
    K = 1
    Z = project_data(X_norm, U, K)
    # 预期值为1.481274
    print(Z[0])

    X_rec = recover_data(Z, U, K)
    # 预期值为-1.047419 -1.047419
    print(X_rec[0, 0], X_rec[0, 1])

    plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
    for i in range(X_norm.shape[0]):
        drawline(X_norm[i, :], X_rec[i, :].reshape(-1, 1), '--k', linewidth=1)
    plt.show()
    input('next step')

    # part4 loading and visualzing face data
    data = io.loadmat('./ex7faces.mat')
    X = data['X']
    plot_data(X[:100, :])

    # part5 pca on face data:eigenfaces
    X_norm, mu, sigma = feature_normalize(X)
    U, S = pca(X_norm)
    plot_data(U[:, :36].T)
    plt.show()
    input('next step')
    plt.close()

    # part6 dimension reduction for faces
    K = 100
    Z = project_data(X_norm, U, K)
    print(Z.shape)

    # part7 visualization of faces after PCA dimension reduction
    K = 100
    X_rec = recover_data(Z, U, K)
    plot_data(X_norm[:100, :])
    plt.title('Origin faces')
    plt.show()

    plt.close()
    plot_data(X_rec[:100, :])
    plt.title('Recovered faces')

    plt.show()
    input('next step')
    plt.close()

    # part8 PCA for visualization
    # (a)
    pic = cv2.imread('./bird_small.png')
    pic = pic / 255
    img_size = pic.shape
    X = pic.reshape(img_size[0] * img_size[1], 3, order='F')
    K = 16
    max_iter = 10
    initial_centroids = Kmeans_randinitial_centroids(X, K)
    centroids, idx = run_Kmeans(X, initial_centroids, max_iter)

    sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int) + 1

    palette = colors.hsv_to_rgb(
        np.column_stack((np.linspace(0, 1, K + 1).reshape(K + 1, -1),
                         np.ones((K + 1, 2)))))
    colors_map = np.array([palette[int(i)] for i in idx[sel]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=colors_map)
    plt.show()

    # (b)
    plt.close()
    fig = plt.figure()
    sel = sel.flatten()
    X_norm, mu, sigma = feature_normalize(X)
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)
    idx = idx.astype(int).reshape(-1, 1)
    plot_data_points(Z[sel, :], idx[sel], K)
    plt.show()
