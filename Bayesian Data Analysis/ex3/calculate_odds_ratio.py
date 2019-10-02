import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def generate_beta(a, b):
    x = np.linspace(0.01, 0.11, 10000)
    y = st.beta.pdf(x, a, b)
    return y


def plot_beta(a, b):
    x = np.linspace(0, 1, 10000)
    y = st.beta.pdf(x, a, b)

    plt.plot(x, y, 'b-')
    plt.title('P1 pdf')
    plt.xlabel('P')
    plt.ylabel('Y')
    plt.show()
    return y


def con_interval(a, b):
    beta = st.beta(a, b)
    return beta.interval(0.95)


def odds_ratio(a1, b1, a2, b2):
    y1 = generate_beta(a1, b1)[2500:7500]
    y2 = generate_beta(a2, b2)[2500:7500]
    y = np.ndarray(5000)
    x = np.linspace(0.01, 0.11, 10000)[2500:7500]
    for i in range(5000):
        y[i] = abs((y2[i] / (1 - y2[i])) / (y1[i] / (1 - y1[i])))
    plt.title('Histogram of odds ratio')
    plt.hist(x, bins=30, weights=y, label='Beta density', alpha=0.5, rwidth=0.9)

    plt.show()
    print('The mean of odds ratio is {}.'.format(y.mean()))
    get_interval(y)


def get_interval(data):
    mean = np.mean(data)
    sigma = np.std(data)

    conf_int = st.norm.interval(0.95, loc=mean, scale=sigma / len(data) ** 0.5)

    print('The interval estimates (0.95) is {}'.format(conf_int))


odds_ratio(40, 655, 23, 678)
