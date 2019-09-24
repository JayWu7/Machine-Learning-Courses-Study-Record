from assignment1.GGL import ggl_generator
from collections import deque
import matplotlib.pyplot as plt


def ran3_generator(x0, amount):
    '''
    RAN3 is a implementation of LFGs
    term:
        LF(55,24,-): xi = (x[i - 55] - x[i-24]) mod m
    '''
    r, s = 55, 24
    m = 2 ** 32
    initial_randoms = deque(ggl_generator(x0, 55))  # using deque to save the memory of store the lags random nums
    for j in range(amount):
        x = initial_randoms[0] - initial_randoms[31]
        # cause we change the lags deque every time, so the index of used random number could be always sure to 0 and 31
        if x > 0:
            x = x % m
        else:
            x = x % -m
        yield x
        initial_randoms.popleft()
        initial_randoms.append(x)


def normalized(rds, m):
    for rd in rds:
        yield abs(rd / m)


def plot_ran3(rds):
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
    plt.scatter(axis_x, axis_y, c='k', s=1.5, alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RAN3 U~(0,1)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    a = ran3_generator(19, 10000)
    b = ran3_generator(19, 10000)
    print(list(a))
    print(list(normalized(b, 2 ** 32)))
    plot_ran3(normalized(ran3_generator(20, 300000), 2 ** 32))
