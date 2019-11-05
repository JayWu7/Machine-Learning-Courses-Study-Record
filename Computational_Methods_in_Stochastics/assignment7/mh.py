import numpy as np
from scipy.stats import gamma
from scipy.stats import expon
import matplotlib.pyplot as plt


def target_distribution(x):
    return gamma.pdf(x, 3, scale=1)


def proposed_distribution(x, lambd):
    return expon.pdf(x, scale=1 / lambd)


def accept_rate(p_x, c_x, lambd):
    '''
    :param p_x: proposed x
    :param c_x: current x
    :return: the accept rate of the proposed x
    '''
    return min(1, target_distribution(p_x) * proposed_distribution(p_x, lambd) / target_distribution(
        c_x)) * p_x


def test_acceptance_rate(lambd, length=10000):
    accetp_num = 0
    # init_x = np.random.exponential(lambd, 1)[0]
    init_x = 0.2
    samples = [init_x]
    for i in range(length):
        p_x = proposed_distribution(samples[-1], lambd)
        rate = accept_rate(p_x, samples[-1], lambd)
        if rate >= np.random.random():
            samples.append(p_x)
            accetp_num += 1
        else:
            samples.append(samples[-1])
    print(accetp_num / length)


def sample(lambd, length=1000):
    init_x = 0.3
    samples = [init_x]
    for _ in range(length):
        p_x = proposed_distribution(samples[-1], lambd)
        rate = accept_rate(p_x, samples[-1], lambd)
        if rate >= np.random.random():
            samples.append(p_x)
        else:
            samples.append(samples[-1])
    return samples


def plot_samples(samples):
    plt.hist(samples, bins=50, rwidth=0.9)
    plt.title('Histogram of Values of x visited by MH algorithm')
    plt.show()


def gamma_plot():
    x = np.linspace(start=0, stop=10, num=100)
    y1 = gamma.pdf(x, a=3, scale=1)
    plt.plot(x, y1)
    plt.title('Gamma distribution')
    plt.show()


gamma_plot()
# test_acceptance_rate(1.63, length=1100)

plot_samples(sample(1.63))
