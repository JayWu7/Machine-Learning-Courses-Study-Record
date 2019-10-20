import numpy as np
import matplotlib.pyplot as plt

lambds = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.9, 7: 0.9, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.6,
          13: 0.6, 14: 0.6, 15: 1.0, 16: 1.0, 17: 0.6, 18: 0.6, 19: 0.5, 20: 0.5, 21: 0.5, 22: 0.5, 23: 0.5}

minutes_whole_day = 24 * 60


def simulate():
    total_time = 0
    times = []
    while total_time < minutes_whole_day:
        l_t = lambds[total_time // 60]
        next_car = np.random.exponential(1 / l_t)
        total_time += next_car
        times.append(total_time)

    y = range(1, len(times) + 1)
    plt.plot(np.array(times) / 60, y)
    plt.title('Simulation of arrival times')
    plt.xlabel('Time')
    plt.ylabel('Number of arrived cars')
    plt.show()


simulate()
