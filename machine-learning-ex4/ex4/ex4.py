import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy import io
from scipy.special import expit


def plot_data(X, example_width=None):
    # ����ͼƬ���δָ����ʹ�þ��������������ŵĳ���
    plt.close()
    plt.figure()
    m, n = X.shape
    if not example_width:
        example_width = np.int32(np.round(np.sqrt(n)))
    # չʾ�Ҷ�ͼ��
    plt.set_cmap('gray')
    example_height = np.int32(n / example_width)

    # ÿ��չʾ���ٺ�ÿ��չʾ����ͼƬ
    display_rows = np.int32(np.floor(np.sqrt(m)))
    display_cols = np.int32(np.ceil(m / display_rows))

    pad = 1
    # ��ʼ�����������ں�����չʾͼƬ
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            # order = 'F'��Ϊ�˱�֤ͼ�������ģ�������ת�õ�
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
    return expit(z)


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def nn_cost_funciton(nn_params,
                     input_layer_size,
                     hidden_layer_size,
                     num_labels,
                     X, y, lambda_):
    m = X.shape[0]
    # ����Ϊ�е�ϵ������ԭ
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))
    # �����������Ԥ����
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = sigmoid(a1@Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), z2))
    h = sigmoid(a2@Theta2.T)

    # �������ʧ����
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
    delta2 = (Theta2@delta3)[:, 1:] * sigmoid_gradient(z2)

    return J


def rand_initial_weights(L_in, L_out):
    epsilon_init = 0.12
    W = random.ranf((L_out, L_in + 1)) * 2 * epsilon_init - epsilon_init
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
    Theta2 = debug_initialize_weights(input_layer_size, num_labels)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.array(range(1, m + 1)) % num_labels
    nn_params = np.concatenate((Theta1.reshape(-1, 1),
                                Theta2.reshape(-1, 1)))
    cost, grad = nn_cost_funciton(nn_params,
                                  input_layer_size,
                                  hidden_layer_size,
                                  num_labels,
                                  X, y, lambda_)
    numgrad = compute_numberical_gradient(nn_cost_funciton, nn_params)
    diff = (numgrad - grad).size / (numgrad + grad).size


def debug_initialize_weights(fan_out, fan_in):
    # ���ջ�ȡ��w��shape
    W_shape = (fan_out, fan_in + 1)
    # ʹ��sin��������ȡ����
    W = np.sin(range(1, fan_out * (fan_in + 1) + 1)).reshape(W_shape) / 10
    return W


if __name__ == '__main__':
    # load the data
    data = io.loadmat('./ex4data1.mat')
    X = data['X']
    y = data['y']

    # plot the numbers
    rand_indices = np.random.permutation(X)
    sel = rand_indices[:100, :]
    # plot_data(sel)

    # part1 load weight and
    weight = io.loadmat('ex4weights.mat')
    Theta1 = weight['Theta1']
    Theta2 = weight['Theta2']

    nn_params = np.concatenate((Theta1.reshape(-1, 1),
                                Theta2.reshape(-1, 1)))

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    # �����������
    lambda_ = 0
    J = nn_cost_funciton(nn_params,
                         input_layer_size,
                         hidden_layer_size,
                         num_labels,
                         X, y, lambda_)
    # Ԥ��ֵΪ0.287629
    print(J)

    # ���������
    lambda_ = 1
    J = nn_cost_funciton(nn_params,
                         input_layer_size,
                         hidden_layer_size,
                         num_labels,
                         X, y, lambda_)
    # Ԥ��ֵΪ0.383770
    print(J)

    initial_Theta1 = rand_initial_weights(hidden_layer_size, input_layer_size)
    initial_Theta2 = rand_initial_weights(hidden_layer_size, num_labels)
    nn_params = np.concatenate((initial_Theta1.reshape(-1, 1),
                                initial_Theta2.reshape(-1, 1)))
