import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_gam(a, b):
    '''
    :param a: gamma shape parameter n
    :param b: gamma scale parameter ℷ
    '''

    x = np.linspace(0, 15, 10000)
    y = stats.gamma.pdf(x, a=a, loc=b)
    plt.hist(x, 20, weights=y, alpha=0.5, rwidth=0.9)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ga(3,2)')
    plt.show()
    print('Sample\'s mean is {}, variance is {}, median is {}.'.format(np.mean(y), np.var(y), np.median(y)))
    return y


def mean_dis(a, b):
    x = np.linspace(0, 15, 10000)
    data = stats.gamma.pdf(x, a=a, loc=b)
    means = np.ndarray((1000,))
    for i in range(1000):
        mean = np.mean(np.random.choice(data, 100))
        means[i] = mean
    x = np.linspace(0, 15, 1000)
    plt.plot(x, means)
    plt.title('Means distribution from samples')
    plt.show()
    print('Variance of the values is {}'.format(np.var(means)))


if __name__ == '__main__':
    mean_dis(3, 2)
