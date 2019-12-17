import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import numpy as np


def read_data(filename, feature):
    data = pd.read_csv(filename, usecols=[feature])
    return data


def get_datas(filenames, feature):
    datas = []
    for fn in filenames:
        datas.append(read_data(fn, feature))
    return datas


def fit_data(datas):
    # plot the histogram
    plt.figure(figsize=(18, 12))
    # first
    plt.subplot(221)
    data = datas[0]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=50, density=True, alpha=0.6, rwidth=0.9)
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=16)
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(15.5, 0.45, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(222)
    data = datas[1]
    data = data[data.SPEED_KNOTSx10 > 3]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=50, density=True, alpha=0.6, rwidth=0.9)
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=16)
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(5.5, 0.19, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(223)
    data = datas[2]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=50, density=True, alpha=0.6, rwidth=0.9)
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=16)
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(24, 0.4, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(224)
    data = datas[3]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=50, density=True, alpha=0.6, rwidth=0.9)
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=16)
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(6.5, 0.3, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.savefig('fig_1.png', dpi=1600)
    # plt.show()


def fit_data_1(datas):
    # plot another feature
    plt.figure(figsize=(18, 12))
    # first
    plt.subplot(221)
    data = datas[0]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=60, density=True, alpha=0.6, rwidth=0.9, color="#E69F00")
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=12, fontweight="bold")
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(1700, 0.0025, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(222)
    data = datas[1]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=60, density=True, alpha=0.6, rwidth=0.9, color="#56B4E9")
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=12, fontweight="bold")
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(1000, 0.004, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(223)
    data = datas[2]
    data = data[data.min_dist < 500]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=60, density=True, alpha=0.6, rwidth=0.9, color="#F0E442")
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=12, fontweight="bold")
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(175, 0.008, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.subplot(224)
    data = datas[3]
    data = data[data.min_dist < 600]
    # data = data[]
    # fit the data
    mu, std = norm.fit(data)
    plt.hist(data.values, bins=60, density=True, alpha=0.6, rwidth=0.9, color="#009E73")
    plt.title('Fit parameters: Mu: {}, Std: {}'.format(round(mu, 3), round(std, 3)), fontsize=12, fontweight="bold")
    # plot the pdf
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=1)
    plt.text(175, 0.01, r'$f(x) = \frac{1}{%s\sqrt{2𝜋}}\cdot{e^{-\frac{(x-%s)^2}{2\cdot{(%s)^2}}}}$' % (
        round(std, 3), round(mu, 3), round(std, 3)), fontsize=15)

    plt.savefig('fit_4.png', dpi=1600)
    # plt.show()


# fit_data(get_datas(['HtT115.csv', 'HtT215.csv', 'TtH115.csv', 'TtH215.csv'], 'SPEED_KNOTSx10'))


def plot_density():
    data = read_data('HtT245.csv', 'min_dist').values
    sns.distplot(data)
    plt.show()

fit_data_1(get_datas(['TtH145.csv', 'HtT145.csv', 'TtH245.csv', 'HtT245.csv'], 'min_dist'))
