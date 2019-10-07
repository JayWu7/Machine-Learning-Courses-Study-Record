'''
implement the Markov Chain to solve the Problem 1
'''

import numpy as np

states = np.linspace(0, 1, 10, endpoint=False)

matrix = np.zeros((10, 10))

# the probability of the state keep in the lowest excited when the current excited is zero is definitely 1
matrix[0][0] = 1

for i in range(1, 9):
    matrix[i][i - 1] = 1 - states[i]
    matrix[i][i + 1] = states[i]

matrix[9][8] = 1 - states[9]
matrix[9][9] = states[9]


def simulate_markov(init_p, times):
    sales_num = 0
    state = int(init_p * 10)
    for _ in range(times):
        if state == 0:
            return sales_num
        elif state == 9:
            prob = [matrix[9][8], matrix[9][9]]
        else:
            prob = [matrix[state][state - 1], matrix[state][state + 1]]

        cur_sale = np.random.choice([0, 1], 1, p=prob)
        if cur_sale:
            sales_num += 1
            # print('from {} to {}, sale'.format(state, state + 1))
            if state < 9:
                state += 1

        else:
            # print('from {} to {}, not sale'.format(state, state - 1))
            state -= 1

    return sales_num


def calculate_average(init_p, markov_times, sample_times):
    total = 0
    for _ in range(sample_times):
        total += simulate_markov(init_p, markov_times)
    return int(total / sample_times)


print(calculate_average(0.3, 500, 10000))
