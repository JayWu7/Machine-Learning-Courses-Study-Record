'''
    GGL is one specific method of LCG( Linear congruential generators)
    LCGs are based on the integer recursion relation:
        x[i+1] =(a * xi +b) mod m, AND to scale to (0,1): x[i+1]/𝑚

    GGL：
        MLCG(16807, 2^31 - 1): x[i+1] = (16807 * xi) mod (2^31 - 1)
'''
import matplotlib.pyplot as plt


def ggl_generator(x0, amount):
    index = 0
    a, b, m = 16807, 0, 2 ** 31 - 1
    while index < amount:
        x = ((a * x0) % m)
        yield x
        x0 = x
        index += 1


def normalized(rds, m):
    for rd in rds:
        yield rd / m


def plot_ggl(rds):
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

    plt.scatter(axis_x, axis_y, c='c', s=1.5, alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GGL U~(0,1)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print(list(ggl_generator(16, 1000)))
    print(list(normalized(ggl_generator(16, 10000), 2 ** 31 - 1)))
    plot_ggl(list(normalized(ggl_generator(16, 100000), 2 ** 31 - 1)))
