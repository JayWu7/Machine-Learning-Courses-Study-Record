'''
plot the beta distribution of posterior probability
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def plot_beta(a, b):
    x = np.linspace(0, 1, 10000)
    y = st.beta.pdf(x, a, b)
    plt.plot(x, y, 'r-', lw=5, alpha=0.6, label='beta pdf')
    plt.show()
    return x, y


def interval_estimate(prob, a, b):
    print('The {} interval estimate is {}'.format(prob, st.beta.interval(prob, a, b)))


def beta_low(prob, a, b):
    print('The probability that the proportion of monitoring sites with detectable algae levels'
          ' π is smaller than {} is {}.'.format(prob, round(st.beta.cdf(prob, a, b), 3)))


def get_mean(a, b):
    print(st.beta.mean(a, b))


if __name__ == '__main__':
    # plot_beta(2, 10)
    interval_estimate(0.95, 74, 380)
    # beta_low(0.19999, 46, 239)
    # print(st.beta.mean(46, 239))
    get_mean(74, 380)
