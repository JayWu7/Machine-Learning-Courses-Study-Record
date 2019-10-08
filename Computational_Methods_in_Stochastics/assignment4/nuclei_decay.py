'''
Let’s take the Poisson process to describe radioactive decay.
The number of nuclei is initially 𝑁& = 10000.
The nuclei decay (fission) at rate 𝜆 = 0.2 per second. We want to determine the half-time 𝑡+/-,
that is, the time it takes on average for the number of nuclei to decay to 𝑁(𝑡) = 𝑁&⁄2.
You can do this simulation in two ways, the first of which is what a statistician would do and that is presented in Lecture 3.
There is an alternative way based by simulating the stochastic process in time steps,
which is what for example a physicist would do.
In this second way you should first run simulations to find appropriate time interval (time step)
but let’s pretend you have already done this and found that
∆𝑡 = 0.01 s. By either simulation, determine the mean and variance for 𝑡+/-.
'''

import numpy as np


def poisson_process(p, n):
    t_interval = 0.01
    l = 0.2  # the parameter ℷ
    p = l * t_interval * np.exp(-l * t_interval)
    decay_num = 0
    time = 0
    while decay_num < n:
        if np.random.choice([0, 1], 1, p=[1 - p, p]):
            decay_num += 1
        time += t_interval
    print(time)
    return time


def test_mean(size):
    times = np.ndarray(size)
    for i in range(size):
        times[i] = poisson_process(0.2, 5000)
    print('Mean is {}, Variance is {}'.format(times.mean(), times.var()))


test_mean(10)
