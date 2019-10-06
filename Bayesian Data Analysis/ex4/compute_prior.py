'''
From the two independent Normal distribution to a joint prior distribution
A = N(0, 2^2), B= N(10, 10^2)
'''

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

custom_styles = dict(
    gray_background={
        'axes.axisbelow': True,
        'axes.edgecolor': 'white',
        'axes.facecolor': '#f0f0f0',
        'axes.grid': True,
        'axes.linewidth': 0.0,
        'grid.color': 'white',
        'xtick.top': False,
        'xtick.bottom': False,
        'ytick.left': False,
        'ytick.right': False,
        'legend.facecolor': 'white'
    }
)

# edit the default plot settings
plt.rc('font', size=12)
# apply custom background plotting style
plt.style.use(custom_styles['gray_background'])


def prior(a, b):
    p = np.exp((-2 / 3) * (a ** 2 / 4 + (b - 10) ** 2 / 100) - a * (b - 10) / 20) / 109
    return p


# calculate the logarithm of the density of the bivariate normal distribution prior
def log_prior(a, b):
    p = np.exp((-2 / 3) * (a ** 2 / 4 + (b - 10) ** 2 / 100) - a * (b - 10) / 20) / 109
    log_p = np.log(p)
    return log_p


# data
x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])


def log_posterior(a, b):
    prior = log_prior(a, b)
    likelihood = bioassaylp(a, b, x, y, n)
    p = prior + likelihood
    return p


def bioassaylp(a, b, x, y, n):
    """Log posterior density for the bioassay problem.
    Given a point(s) and the data, returns unnormalized log posterior density
    for the bioassay problem assuming uniform prior.
    Parameters
    ----------
    a, b : scalar or ndarray
        The point(s) (alpha, beta) in which the posterior is evaluated at.
        `a` and `b` must be of broadcastable shape.
    x, y, n : ndarray
        the data vectors
    Returns
    -------
    lp : scalar or ndarray
        the log posterior density at (a, b)
    """
    # last axis for the data points
    a = np.expand_dims(a, axis=-1)
    b = np.expand_dims(b, axis=-1)
    # these help using chain rule in derivation
    t = a + b * x
    et = np.exp(t)
    z = et / (1. + et)
    eps = 1e-12
    z = np.minimum(z, 1 - eps)
    z = np.maximum(z, eps)
    # negative log posterior (error function to be minimized)
    lp = np.sum(y * np.log(z) + (n - y) * np.log(1.0 - z), axis=-1)
    return lp


def plot_posterior():
    A = np.linspace(-4, 4, 1000)
    B = np.linspace(-10, 30, 1000)

    ilogit_abx = 1 / (np.exp(-(A[:, None] + B[:, None, None] * x)) + 1)
    p = np.prod(ilogit_abx ** y * (1 - ilogit_abx) ** (n - y), axis=2)

    rng = np.random.RandomState(0)
    nsamp = 1000
    samp_indices = np.unravel_index(
        rng.choice(p.size, size=nsamp, p=p.ravel() / np.sum(p)),
        p.shape
    )
    samp_A = A[samp_indices[1]]
    samp_B = B[samp_indices[0]]
    # add random jitter, see BDA3 p. 76
    samp_A += (rng.rand(nsamp) - 0.5) * (A[1] - A[0])
    samp_B += (rng.rand(nsamp) - 0.5) * (B[1] - B[0])

    # samples of LD50 conditional beta > 0
    bpi = samp_B > 0
    samp_ld50 = -samp_A[bpi] / samp_B[bpi]

    # create figure
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # plot the posterior density
    ax = axes[0]
    ax.imshow(p, origin='lower', aspect='auto', extent=(A[0], A[-1], B[0], B[-1]))

    ax.set_xlim([-2, 8])
    ax.set_ylim([-2, 40])
    ax.set_ylabel(r'$\beta$')
    ax.grid(True)
    ax.set_title('posterior density')

    # plot the samples
    ax = axes[1]
    ax.scatter(samp_A, samp_B, 10, linewidth=0)
    ax.set_xlim([-2, 8])
    ax.set_ylim([-2, 40])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title('samples')

    fig.tight_layout()

    plt.show()


def weight(a, b):
    q = np.exp(log_posterior(a, b))
    g = prior(a, b)
    return q / g


def log_weight(a, b):
    return np.log(weight(a, b))


def normalized_weight(weights):
    new_weights = np.empty(len(weights))
    for i in range(len(weights)):
        new_weights[i] = np.exp(weights[i])
    new_weights = new_weights / sum(new_weights)
    return new_weights


def estimate_mean(size):
    sam_a = st.norm.rvs(0, 2, size)
    sam_b = st.norm.rvs(10, 10, size)

    numerator = np.empty(2)
    denominator = 0
    weights = []
    for a, b in zip(sam_a, sam_b):
        w = weight(a, b)
        weights.append(w)
        denominator += w
        numerator[0] += a * w
        numerator[1] += b * w

    print(numerator / denominator)
    return weights


def compute_s_eff(weights):
    nor_weights = normalized_weight(weights)
    s_eff = 0
    for w in nor_weights:
        s_eff += (w ** 2)
    return 1 / s_eff


def generate_samples(size):
    sam_a = st.norm.rvs(0, 2, size)
    sam_b = st.norm.rvs(10, 10, size)
    weights = np.ndarray(size)
    samples = np.ndarray(size)

    for i in range(size):
        weights[i] = weight(sam_a[i], sam_b[i])
        samples[i] = log_posterior(sam_a[i], sam_b[i])

    weights = normalized_weight(weights)
    return weights, samples, sam_a, sam_b


def generate_resample(weights, samples, samp_a, samp_b, size):
    resamples = np.ndarray(size)
    resamples_b = np.ndarray(size)
    resamples_a = np.ndarray(size)
    indexes = range(len(samples))
    for i in range(size):
        index = np.random.choice(indexes, p=weights, replace=False)
        resamples[i] = samples[index]
        resamples_b[i] = samp_b[index]
        resamples_a[i] = samp_a[index]

    return resamples, resamples_a, resamples_b, indexes


def plot_importance_resampling(resamples, resamples_a, resamples_b):
    plt.scatter(range(len(resamples)), resamples, alpha=0.6)
    plt.title('Resampling posterior scatter')
    plt.ylabel('Log posterior')
    plt.show()
    plt.scatter(resamples_a, resamples_b, alpha=0.6)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.show()


def estimate_p(resampling_b):
    res = 0
    for s in resampling_b:
        if s > 0:
            res += 1
    return res / len(resampling_b)


def plot_hist_posterior(resamples_a, resamples_b):
    new_resample = []
    for i in range(len(resamples)):
        if resamples_b[i] > 0:
            new_resample.append(-resamples_a[i] / resamples_b[i])
    new_resamples = np.array(new_resample)

    plt.hist(new_resamples, bins=30, color='y', rwidth=0.9, range=(-1, 1), alpha=0.5)
    plt.title('LD50')
    plt.show()


if __name__ == '__main__':
    # samples_a = [1.896, -3.6, 0.374, 0.964, -3.123, -1.581]
    # samples_b = [24.76, 20.04, 6.15, 18.65, 8.16, 17.4]
    #
    # weights = []
    # for a, b in zip(samples_a, samples_b):
    #     weights.append(log_weight(a, b))
    # print(weights)
    # print(normalized_weight(weights))
    # print(sum(normalized_weight(weights)))
    # print(estimate_mean(1000))
    # print(compute_s_eff(estimate_mean(100)))
    weights, samples, sam_a, sam_b = generate_samples(10000)
    resamples, resamples_a, resamples_b, indexes = generate_resample(weights, samples, sam_a, sam_b, 1000)
    # plot_importance_resampling(resamples, resamples_a, resamples_b)
    # print(estimate_p(resamples_b))
    plot_hist_posterior(resamples_a, resamples_b)
# plot_posterior()
