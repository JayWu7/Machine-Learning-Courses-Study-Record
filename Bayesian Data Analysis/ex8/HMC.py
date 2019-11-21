import numpy as np
import matplotlib.pyplot as plt


def compute_U(q):
    pdf = 2 * np.exp(-2 * q)
    u = -np.log(pdf)
    return u


def grad_U(q):
    return 2


def HMC(cur_q, epsilon=0.01, L=25):
    q = cur_q
    p = float(np.random.normal(0, 1, size=1))
    cur_p = p
    p = p - epsilon * grad_U(q) / 2
    for i in range(1, L + 1):
        q = q + epsilon * p
        if i != L:
            p = p - epsilon * grad_U(q)

    p = p - epsilon * grad_U(q) / 2
    p = -p

    if q < 0 and p > 0:
        q = -q
    if q < 0 and p < 0:
        q = -q
        p = -p

    cur_u = compute_U(cur_q)
    cur_k = cur_p ** 2 / 2
    proposed_u = compute_U(q)
    proposed_k = p ** 2 / 2
    u = np.random.random()
    if u < np.exp(cur_u - proposed_u + cur_k - proposed_k):
        return q
    else:
        return cur_q


def plot_dist(data):
    plt.title('Distribution of samples')
    plt.hist(data, bins=50, rwidth=0.9, color='black')
    plt.show()


if __name__ == '__main__':
    samples = np.ndarray(10000)
    q = 1
    for i in range(10000):
        q = HMC(q)
        samples[i] = q
    print('mean: ', np.mean(samples))
    print('var: ', np.var(samples))

    plot_dist(samples)
