import numpy as np
import matplotlib.pyplot as plt

n = 20
mean = 30
variance = 20
a = 1
b = 1
c = 3
d = 50


def gibbs_sampler(size):
    mu_set = np.ndarray(size)
    tau_set = np.ndarray(size)
    mu, tau = 3, 1  # initial values of mu and tau
    para_a = a + n / 2
    for i in range(size):
        para_b = b + ((n - 1) * variance + n * ((mean - mu) ** 2)) / 2
        tau = np.random.gamma(para_a, para_b)
        para_c = (c * d + n * tau * mean) / (n * tau + d)
        para_d = 1 / (n * tau + d)
        mu = np.random.normal(para_c, np.sqrt(para_d))

        mu_set[i] = mu
        tau_set[i] = tau

    return mu_set, tau_set


samples = gibbs_sampler(10000)


def plot_joint(samples):
    index = [i for i in range(len(samples[0])) if 29.94 < samples[0][i] < 30.01 and samples[1][i] < 5000]
    x = [samples[0][i] for i in index]
    y = [samples[1][i] for i in index]
    plt.scatter(x, y, alpha=0.02, color='black', linewidths=0.5)
    plt.title('Joint distribution for mu and tau')
    plt.xlabel('mu')
    plt.ylabel('tau')
    plt.show()


def plot_marginal_mu(samples):
    index = [i for i in range(len(samples[0])) if 29.94 < samples[0][i] < 30.01 and samples[1][i] < 5000]
    x = [samples[0][i] for i in index]
    plt.hist(x, bins=60, rwidth=0.8)
    plt.title('Marginal distribution for mu')
    plt.xlabel('mu')
    plt.show()


def plot_marginal_tau(samples):
    index = [i for i in range(len(samples[0])) if 29.94 < samples[0][i] < 30.01 and samples[1][i] < 5000]
    x = [samples[1][i] for i in index]
    plt.hist(x, bins=60, rwidth=0.9)
    plt.title('Marginal distribution for tau')
    plt.xlabel('tau')
    plt.show()


def plot_marginal_variance(samples):
    index = [i for i in range(len(samples[0])) if 29.94 < samples[0][i] < 30.01 and samples[1][i] < 5000]
    x = [1/samples[0][i] for i in index]
    plt.hist(x, bins=60, rwidth=0.9)
    plt.title('Marginal distribution for variance')
    plt.xlabel('variance')
    plt.show()

# plot_joint(samples)
plot_marginal_mu(samples)
# plot_marginal_tau(samples)
# plot_marginal_variance(samples)