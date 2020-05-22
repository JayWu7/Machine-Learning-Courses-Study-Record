def iteration(number):
    v0 = v1 = 0
    y = 0.9
    for i in range(number):
        cur_v0 = max(1.0 * (1 + y * v1), 1.0 * (0 + y * v0))
        cur_v1 = max(0.1 * (0 + y * v1) + 0.9 * (0 + y * v0), 1.0 * (0 + y * v1))

        v0 = cur_v0
        v1 = cur_v1

        print('Iteration {}:'.format(i))
        print(v0)
        print(v1)


iteration(3)
