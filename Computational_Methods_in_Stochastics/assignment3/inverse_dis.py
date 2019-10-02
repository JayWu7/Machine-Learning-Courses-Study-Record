import numpy as np
import matplotlib.pyplot as plt


def generate_u(size, low=0, high=19263):
    return np.random.uniform(low, high, size)


def generate_m(size):
    samples_u = generate_u(size)
    samples_m = np.ndarray(size)

    for i, u in enumerate(samples_u):
        samples_m[i] = 400000000 / (u ** 2 - 40000 * u + 400000000)

    print(samples_m)
    return samples_m


def linear_binning_plot(data):
    plt.hist(data, bins=20, alpha=0.5, rwidth=0.9)
    plt.title('Histogram of M with linear binning')
    plt.xlabel('M')
    plt.show()


def log_binning_plot(data):
    data = np.log(data)
    plt.hist(data, bins=20, alpha=0.5, rwidth=0.9)
    plt.title('Histogram of M with logarithmic binning')
    plt.xlabel('M')
    plt.show()


if __name__ == '__main__':
    data = generate_m(100)
    linear_binning_plot(data)
    log_binning_plot(data)
