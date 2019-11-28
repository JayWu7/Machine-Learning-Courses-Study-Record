import numpy as np

matrix = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
vector1 = np.array([[0.1, 0.6, 0.3]], dtype=float)

for i in range(100):
    matrix = np.matmul(matrix, matrix)
    print("Current round:", i + 1)
    # print(vector1)
    print(matrix)

# matrix = np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
# for i in range(10):
#     matrix = np.matmul(matrix, matrix)
#     print("Current round:", i + 1)
#     print(matrix)


# Why the stable distribution of matrix changed when n get bigger.
