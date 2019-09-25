import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def generate_y():
    for u in np.random.random_sample(10000):
        yield -np.log(1 - u) / 2 - 1  # yield y


def plot_y():
    y = list(generate_y())

    plt.hist(y, bins=50, alpha=0.5, rwidth=0.9, color='r')
    plt.title('Distribution of Y')
    plt.xlabel('Y')
    plt.ylabel('Quantity')

    plt.show()


def generate_x():
    x = np.random.exponential(1, 10000)
    return x


def plot_x():
    x = generate_x()
    plt.hist(x, bins=50, alpha=0.5, rwidth=0.9, color='y')
    plt.title('Distribution of X')
    plt.xlabel('X')
    plt.ylabel('Quantity')

    plt.show()


def simulate_z():  # z = x + y
    x = generate_x()
    y = list(generate_y())
    return x + y


def plot_z():
    z = simulate_z()
    plt.hist(z, bins=50, alpha=0.5, rwidth=0.8, color='g')
    plt.title('Distribution of Z')
    plt.xlabel('Z')
    plt.ylabel('Quantity')

    plt.show()


def pdf_z():
    axis = np.linspace(0, 1, 10000, endpoint=False)
    x = [np.exp(-i) for i in axis]

    y = [(-np.log(1 - u) / 2 - 1) for u in axis]

    x = np.array(x)
    y = np.array(y)

    z = x + y

    plt.plot(axis, z, 'k')
    plt.title('PDF of Z')
    plt.xlabel('U')
    plt.ylabel('Z')

    plt.show()


if __name__ == '__main__':
    # plot_y()
    # plot_x()
    # print(simulate_z())
    # plot_z()

    pdf_z()
