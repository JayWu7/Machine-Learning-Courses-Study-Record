import numpy as np

a = [1, 6, 4, 2, 6, 4, 1]
b = [1, 5, 9]
c = [6, 4, 1]

# a = 3,3,3,2,2,2,1,1,1
# b = 3,3,2,2
# c = [4]

data = [a, b, c]


def u1b1(data):
    n = len(data)
    index_n = n
    values = [d[0] + np.sqrt(2 * np.log(n)) for d in data]
    compare_values = [[values[i], i] for i in range(n)]
    already_calculate = [[data[i][0]] for i in range(n)]
    lengths = [len(d) for d in data]
    maxn = sum(lengths)
    data = [d[1:] for d in data]

    while n < maxn:
        n += 1
        i = max(compare_values)[1]
        for k in range(index_n):
            if i == k:
                already_calculate[k].append(data[k][0])
                data[k] = data[k][1:]
                if not data[k]:
                    compare_values[k][0] = -float('inf')
            values[k] = np.mean(already_calculate[k]) + np.sqrt(2*np.log(n) / len(already_calculate[k]))
            if compare_values[k][0] != -float('inf'):
                compare_values[k][0] = values[k]

    print(values)

u1b1(data)



