import numpy as np
import matplotlib.pylot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def loaddata(filename):
    data = loadmat(filename)
    return data['X'], data['y']


if __name__ == '__main__':
