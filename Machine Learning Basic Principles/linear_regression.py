import numpy as np

x = np.array([0, 1, 3, 4, 6, 8, 9])
y = np.array([2, 5, 7, 6, 8, 14, 12])

x_mean = x.mean()
y_mean = y.mean()
up = 0
bo_l = 0
bo_r = 0
m = len(x)

for i in range(m):
    up += y[i] * (x[i] - x_mean)
    bo_l += x[i] ** 2
    bo_r += x[i]

w = up / (bo_l - bo_r ** 2 / m)

va = 0
for i in range(m):
    va += (y[i] - w * x[i])

b = va / m

# cov_xy = np.cov(x, y)
# print(cov_xy)
# v_x = np.var(x)
# print(v_x)

# print(w, b)
# print(cov_xy / v_x)

cov, v_x = 0, 0
for i in range(m):
    cov += (x[i] - x_mean) * (y[i] - y_mean)
    v_x += (x[i] - x_mean) ** 2

print(cov / v_x)

# Be careful that using numpy to compute the variance and cov_variance is little different than using the formula
