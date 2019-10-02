import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def read_data():
    '''
    read data from the windshieldy1.txt
    :return: mean of samples,
    '''

    with open('./windshieldy1.txt', 'r') as f:
        data = [float(line[:-1]) for line in f.readlines()]  # reduce the final \n of each line.
        n = len(data)
        y = np.mean(data)
        res = 0
        for d in data:
            res += (d - y) ** 2
        s = res / (n - 1)
        return y, s, n


def t_interval(a, df, loc, scale):
    interval = st.t.interval(a, df, loc, scale)
    print('The {} interval estimate is {}.'.format(a, interval))


def t_pdf(df, a, b):
    x = np.linspace(st.t.ppf(0.01, df, a, b), st.t.ppf(0.99, df, a, b), 100)
    plt.plot(x, st.t.pdf(x, df, a, b), 'r')
    plt.title('μ pdf')
    plt.show()


def get_ud(a1, b1, df1, a2, b2, df2):
    x1 = np.linspace(st.t.ppf(0.01, df1, a1, b1), st.t.ppf(0.99, df1, a1, b1), 100)
    x2 = np.linspace(st.t.ppf(0.01, df2, a2, b2), st.t.ppf(0.99, df2, a2, b2), 100)

    y1 = st.t.pdf(x1, df1, a1, b1)
    y2 = st.t.pdf(x2, df2, a2, b2)

    y = y1 - y2

    plt.hist(x1, bins=30, weights=y, alpha=0.5, rwidth=0.9)
    plt.title('Histogram of ud')
    plt.show()
    print('The mean of ud is {}.'.format(round(y.mean(), 3)))
    get_interval(y)


def get_interval(data):
    mean = np.mean(data)
    sigma = np.std(data)

    conf_int = st.norm.interval(0.95, loc=mean, scale=sigma / len(data) ** 0.5)

    print('The interval estimates (0.95) is {}'.format(conf_int))


get_ud(14.6, 0.24, 8, 15.82, 0.05, 12)
