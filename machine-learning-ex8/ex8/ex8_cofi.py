from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.optimize import minimize


def cofi_costfunc(params, Y, R, num_users, num_movies,
                  num_features, lambda_):
    # 还原参数
    X = params[:num_movies *
               num_features].reshape(num_movies, num_features, order='F')
    Theta = params[num_movies *
                   num_features:].reshape(num_users, num_features, order='F')
    # [num_movies,num_users]
    h = X@Theta.T
    # 协同滤波算法代价函数
    J = ((R * (h - Y)**2).sum() + lambda_ *
         (Theta**2).sum() + lambda_ * (X**2).sum()) / 2
    X_grad = (R * (h - Y))@Theta + lambda_ * X
    Theta_grad = (R * (h - Y)).T@X + lambda_ * Theta
    grad = np.vstack((
        X_grad.reshape(-1, 1, order='F'),
        Theta_grad.reshape(-1, 1, order='F')
    ))
    return J, grad


def check_cost_function(lambda_=0):
    # 创建小的测试数据
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # 将不符合的数据去除掉
    Y = X_t@Theta_t.T
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 1
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # 进行梯度检验
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    params = np.vstack((
        X.reshape(-1, 1, order='F'),
        Theta.reshape(-1, 1, order='F')))

    def cost_func(p):
        return cofi_costfunc(p, Y, R, num_users, num_movies,
                             num_features, lambda_)
    numgrad = compute_numberical_gradient(cost_func, params)
    _, grad = cofi_costfunc(params, Y, R, num_users, num_movies,
                            num_features, lambda_)
    for pair in zip(numgrad, grad):
        print(pair)

    # 科学记数法显示误差
    diff = Decimal(np.linalg.norm(numgrad - grad)) / \
        Decimal(np.linalg.norm(numgrad + grad))
    # 应该小于1e-9
    print(diff)


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


def load_movie_list():
    movie_list = dict()
    with open('./movie_ids.txt', 'r', encoding='utf8') as fp:
        i = 1
        for line in fp.readlines():
            line = line.strip()
            index = line.index(' ')
            idx, movie_name = line[:index], line[index + 1:]
            movie_list[int(idx) - 1] = movie_name
    return movie_list


def normalize_rating(Y, R):
    m, n = Y.shape
    Y_mean = np.zeros((m, 1))
    Y_norm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Y_mean[i] = np.mean(Y[i, idx])
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]
    return Y_norm, Y_mean


if __name__ == '__main__':
    # part1 load movie rating dataset
    data = io.loadmat('./ex8_movies.mat')
    Y = data['Y']
    R = data['R']
    # 预期值为3.8783
    print(np.mean(Y[0, R[0, :] == 1]))
    plt.imshow(Y)
    plt.ylabel('Movies')
    plt.xlabel('Users')
    # plt.show()
    # input('next step')
    plt.close()

    # part2 collaboative filtering cost function
    data = io.loadmat('./ex8_movieParams.mat')
    X = data['X']
    Theta = data['Theta']

    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]

    params = np.vstack((
        X.reshape(-1, 1, order='F'),
        Theta.reshape(-1, 1, order='F')
    ))
    J, grad = cofi_costfunc(params, Y, R, num_users,
                            num_movies, num_features, 0)
    # 预期值为22.22
    # print(J)
    # input('next step')

    # part3 collaborative filtering gradient
    # check_cost_function()
    # input('next step')

    # part4 collaborative filtering cost reularization
    J, _ = cofi_costfunc(params, Y, R, num_users, num_movies,
                         num_features, 1.5)
    # 预期值为31.34
    # print(J)
    # input('next step')

    # part5 collaborative filtering gradient regularization
    # check_cost_function(1.5)
    # input('next step')

    # part6 entering rating for a new user
    movie_list = load_movie_list()
    my_ratings = np.zeros((1682, 1))
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[66] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    # for i in range(len(my_ratings)):
    #     if my_ratings[i] > 0:
    #         print('Rated {} for {}'.format(
    #             int(my_ratings[i]), movie_list[i]))
    # input('next step')

    # part7
    data = io.loadmat('./ex8_movies.mat')
    Y = data['Y']
    R = data['R']
    Y = np.hstack((my_ratings, Y))
    R = np.hstack((np.array(my_ratings != 0), R))
    Y_norm, Y_mean = normalize_rating(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)

    initial_parameters = np.vstack((
        X.reshape(-1, 1, order='F'),
        Theta.reshape(-1, 1, order='F')
    ))
    lambda_ = 10

    def cost_func(p):
        return cofi_costfunc(p, Y_norm, R, num_users, num_movies,
                             num_features, lambda_)
    myoptions = {'maxiter': 100, 'disp': False}
    result = minimize(cost_func, initial_parameters,
                      method='L-BFGS-B', options=myoptions, jac=True)
    theta = result['x']
    X = theta[:num_movies * num_features].reshape(
        num_movies, num_features, order='F')
    Theta = theta[num_movies * num_features:].reshape(
        num_users, num_features, order='F')

    # part8 recommendation for you
    p = X@Theta.T
    my_prediction = p[:, 0] + Y_mean.flatten()
    movie_list = load_movie_list()
    ix = my_prediction.argsort()[::-1]
    print('\nTop recommendations for you:\n')
    for i in range(10):
        j = ix[i]
        print(j)
        print('Predicting rating {} for movie {}'.format(
            my_prediction[j], movie_list[j]))

    print('\n\nOriginal ratings provided:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated {} for {}'.format(
                my_ratings[i], movie_list[i]))
