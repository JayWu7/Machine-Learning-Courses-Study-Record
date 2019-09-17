import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

mean = 0.2
variance = 0.01


def calculate_a(mean, variance):
    return mean * (mean * (1 - mean) / variance - 1)


def calculate_b(a, mean):
    return a * (1 - mean) / mean


def plot_beta(a, b):
    x = np.linspace(0, 1, 10000)
    y = st.beta.pdf(x, a, b)
    plt.plot(x, y, 'r-', lw=5, alpha=0.6, label='beta pdf')
    plt.show()
    return x, y


def plot_histogram(data, size, a):
    indexs = np.random.choice(range(10000), size)
    x = np.array([data[0][i] for i in indexs])
    y = np.array([data[1][i] for i in indexs])
    plt.hist(x, bins=30, weights=y, label='Beta density', alpha=0.5, rwidth=0.9)
    plt.show()
    mean = sum(x * y) / 1000  # calculate the mean of sample
    var = mean * (1 - mean) / (1 + (a / mean))
    print('Drawn samples mean is: {}, var is {}'.format(mean, var))
    con_interval = calculate_interval(x * y)
    print('Estimation of the central 95%-interval of the distribution is {}'.format(con_interval))

    con_interval = st.beta.ppf()


def calculate_interval(s):
    return st.t.interval(0.95, len(s) - 1, loc=np.mean(s), scale=st.sem(s))




if __name__ == '__main__':
    a = calculate_a(mean, variance)
    b = calculate_b(a, mean)
    data = plot_beta(a, b)
    plot_histogram(data, 1000, a)


