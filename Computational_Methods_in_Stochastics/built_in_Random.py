from random import random
import numpy as np
import matplotlib.pyplot as plt


def generate_random(n):
    for _ in range(n):
        yield random()


def plot_mt(rds):
    axis_x, axis_y = [], []
    try:
        while True:
            axis_x.append(next(rds))  # iterate the randoms generator
            axis_y.append(next(rds))
    except StopIteration:
        pass
    length = len(axis_x)
    interval_length = int(length * (10 ** -3))
    axis_x = axis_x[length // 2:(length // 2 + interval_length)]
    axis_y = axis_y[length // 2:(length // 2 + interval_length)]
    plt.scatter(axis_x, axis_y, s=1.5, alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mersenne Twister U~(0,1)')
    plt.grid(True)
    plt.legend()
    plt.show()

np.va

if __name__ == '__main__':
    print(list(generate_random(1000)))
    plot_mt(list(generate_random(100000)))
