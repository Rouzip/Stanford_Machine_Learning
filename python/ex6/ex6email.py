import re

from nltk import PorterStemmer
from sklearn.svm import SVC
from scipy import io
import numpy as np


def load_txt(filename):
    with open(filename, 'r') as fp:
        data = fp.read()
        if data == '':
            print('无内容')
    return data


def get_vacabulary_list(filename):
    voca_list = dict()
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            lists = line.split()
            index = int(lists[0])
            vocab = lists[-1].strip()
            voca_list[vocab] = index
    return voca_list


def process_email(email_contents):
    voca_list = get_vacabulary_list('./vocab.txt')
    word_indices = list()
    email_contents = email_contents.lower()
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)
    email_contents = re.sub(r'[0-9]+', ' ', email_contents)
    email_contents = re.sub(
        r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)
    print('\n==== Processed Email ====\n\n')

    # python和matlab正则不同，有些符号需要加上转义
    email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+',
                              email_contents)
    for vocabulary in email_contents:
        vocabulary = re.sub(r'[^a-zA-Z0-9]', '', vocabulary)
        vocabulary = PorterStemmer().stem(vocabulary.strip())

        if len(vocabulary) <= 1:
            continue

        # 将对应单词的位置加入list
        if vocabulary in voca_list:
            index = voca_list[vocabulary]
            word_indices.append(index)
        else:
            index = 0
    print('\n\n=========================\n')

    return word_indices


def email_feature(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    for index in word_indices:
        x[index - 1] = 1
    return x


if __name__ == '__main__':
    # part1 email数据处理
    file_contents = load_txt('./emailSample1.txt')
    word_indices = process_email(file_contents)
    print(word_indices)
    input('next step')

    # part2 特征工程
    file_contents = load_txt('./emailSample1.txt')
    word_indices = process_email(file_contents)
    features = email_feature(word_indices)

    print('Length of feature vector: {}\n'.format(len(features)))
    print('Number of non-zero entries: {}\n'.format(np.sum(features > 0)))
    input('next step')

    # part3 测试垃圾邮件分类
    data = io.loadmat('./spamTrain.mat')
    X = data['X']
    y = data['y'].flatten()
    C = 0.1
    model = SVC(C=C, kernel='linear')
    model.fit(X, y)
    result = model.predict(X)
    print('Training Accuracy: {}\n'.format(np.mean(result == y) * 100))
    input('next step')

    # part4 测试垃圾邮件分类
    data = io.loadmat('spamTest.mat')
    Xtest = data['Xtest']
    ytest = data['ytest'].flatten()
    result = model.predict(Xtest)
    print('Training Accuracy: {}\n'.format(np.mean(result == ytest) * 100))
    input('next step')

    # part5 预测垃圾邮件关键词
    weight = model.coef_[0]
    voca_list = sorted(get_vacabulary_list('./vocab.txt').keys())
    indices = weight.argsort()[::-1][:15]
    print('\nTop predictors of spam: \n')
    for index in indices:
        print(voca_list[index], weight[index], '\n')
    input('next step')

    # part6 尝试自己的邮件
    filename = input('your filename: ')
    file_contents = load_txt(filename)
    word_indices = process_email(file_contents)
    X = email_feature(word_indices).reshape(1, -1)
    p = model.predict(X)
    print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, p))
    print('(1 indicates spam, 0 indicates not spam)\n\n')
