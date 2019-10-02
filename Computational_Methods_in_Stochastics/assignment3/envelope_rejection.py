'''
implement the envelop rejection method
'''

import numpy as np


def h_x(x):
    '''
    using the  h(𝑥) = e^-x
    '''
    return np.exp(-x)


def u_a(a, y):
    '''
    generate the u of U(0,ah(x))
    :return:
    '''
    return np.random.uniform(0, a * h_x(y))


def rejection(u, fy):
    '''
    :param u:
    :param f_y: f(y)
    :return: if u is less than f(y)
    '''
    return u < fy


def f_y(y):
    '''
    we using the density function of N(0,1) as f(y)
    :return:
    '''
    return np.exp((-y ** 2 / 2)) / np.sqrt(2 * np.pi)


def envelop(a):
    x = np.random.randint(0, 10000, 1)
    y = h_x(x)
    u = u_a(a, y)
    fy = f_y(y)
    if rejection(u, fy):
        return y
    else:
        return False


def change_sign(samples):
    return -samples


def main(size, a):
    '''
    :param size: the amount of  sample n
    :param a: is an upper bound for 𝑓(𝑥)/h(𝑥)
    :return:
    '''
    samples = []
    for i in range(size):
        y = envelop(a)
        if y:
            samples.append(y)
    samples = np.array(samples)
    return samples


def adding(ne_s, po_s):
    return np.append(ne_s, po_s)


def rescale(a, b, s):
    return a * s + b


if __name__ == '__main__':
    s = main(100000, 1.2)
    negative_s = change_sign(s)
    new_s = rescale(2, 2, adding(negative_s, s))
    print(new_s[:10])
    print(new_s.shape)
    print(new_s.mean(), s.var())
