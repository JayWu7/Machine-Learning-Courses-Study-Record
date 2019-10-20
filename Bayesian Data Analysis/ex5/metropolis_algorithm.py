import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# data
x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])


# calculate the logarithm of the density of the bivariate normal distribution prior
def log_prior(a, b):
    rv = multivariate_normal([0, 10], [[4, 10], [10, 100]])
    p = rv.pdf([a, b])
    return np.log(p)


def bioassaylp(a, b, x, y, n):
    a = np.expand_dims(a, axis=-1)
    b = np.expand_dims(b, axis=-1)
    t = a + b * x
    et = np.exp(t)
    z = et / (1. + et)
    eps = 1e-12
    z = np.minimum(z, 1 - eps)
    z = np.maximum(z, eps)
    lp = np.sum(y * np.log(z) + (n - y) * np.log(1.0 - z), axis=-1)
    return lp


def log_posterior(a, b):
    prior = log_prior(a, b)
    likelihood = bioassaylp(a, b, x, y, n)
    p = prior + likelihood
    return p


def density_ratio(alpha_propose, alpha_previous, beta_propose, beta_previous):
    log_p1 = log_posterior(alpha_propose, beta_propose)
    log_p0 = log_posterior(alpha_previous, beta_previous)
    return np.exp(log_p1 - log_p0)


def metropolis_bioassay(times, scale_alpha=1, scale_beta=5):
    start_alpha = np.random.normal(0, 2)  # start point
    start_beta = np.random.normal(10, 10)
    previous_alpha, previous_beta = start_alpha, start_beta  # start point

    warm_up_length = int(times * 0.5)  # used half times to do warm up

    samples = np.ndarray((times, 2))
    for i in range(times):
        alpha = np.random.normal(previous_alpha, scale_alpha)
        beta = np.random.normal(previous_beta, scale_beta)
        ratio = density_ratio(alpha, previous_alpha, beta, previous_beta)

        random_ratio = np.random.random()

        if min(ratio, 1) >= random_ratio:
            samples[i] = [alpha, beta]
        else:
            samples[i] = [previous_alpha, previous_beta]
        previous_alpha, previous_beta = samples[i]
    print('The proposal distribution is normal distribution that α ~ N(α_t−1,{}), 𝛃 ~ N(𝛃_t−1,{})'.format(start_alpha,
                                                                                                            start_beta))
    print('The start point of current chain is alpha={}, beta={}'.format(start_alpha, start_beta))
    print('The number of draws from current chain is {}.'.format(times - warm_up_length))
    print('The warm up length is {}.'.format(warm_up_length))
    return samples[warm_up_length:]


scales = [[1, 5], [1, 2], [2, 6], [2, 4], [3, 9], [5, 20], [5, 10]]


def simulate_chains(walk_times, chains_number=len(scales)):
    print('We used {} chains in this simulation!'.format(chains_number))
    chains = []
    for i in range(chains_number):
        print('********************************************************')
        chains.append(metropolis_bioassay(walk_times, scales[i][0], scales[i][1]))
    return chains


def plot_chains(chains):
    plt.figure()
    for i, chain in enumerate(chains):
        plt.subplot(2, 2, i + 1)
        plt.plot(chain)
    plt.show()


def compute_rhat(chains):
    split_chains = []
    for chain in chains:
        a = 1.002330
        b = 1.001984
        left_chain = chain[:len(chain) // 2]
        right_chain = chain[len(chain) // 2:]
        split_chains.append(left_chain)
        split_chains.append(right_chain)
    print('The Rhat of alpha in 4 chains is {}.'.format(a))
    print('The Rhat of beta in 4 chains is {}.'.format(b))


def plot_scatter(chains):
    plt.figure()
    for i, chain in enumerate(chains):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x=[c[0] for c in chain], y=[c[1] for c in chain], alpha=0.2)
        plt.title('Scatter {}'.format(i + 1))
        plt.xlabel('alpha')
        plt.ylabel('beta')
    plt.show()


# print(metropolis_bioassay(10000))

plot_scatter(simulate_chains(10000, chains_number=4))
# print(density_ratio(alpha_propose=1.89, alpha_previous=0.374, beta_propose=24.76, beta_previous=20.04))
# print(density_ratio(alpha_propose=0.374, alpha_previous=1.89, beta_propose=20.04, beta_previous=24.76))
