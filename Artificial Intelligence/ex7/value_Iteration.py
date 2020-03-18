

def iteration(number):
    v1 = v2 = v3 = 0
    y = 0.9
    for i in range(number):
        cur_v1 = 1.0*(0 + y * v2)
        cur_v2 = 1.0*(0 + y * v3)
        cur_v3 = (0.2*(10 + y*v1) + 0.8*(0 + y * v2))

        v1 = cur_v1
        v2 = cur_v2
        v3 = cur_v3

        print('Iteration {}:'.format(i))
        print(v1)
        print(v2)
        print(v3)

iteration(500)