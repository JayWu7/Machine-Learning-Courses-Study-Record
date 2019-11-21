import numpy as np

a = [3, 0, 3, 0]
b = [0, 6, 0, 6]
c = [1, 1, 1, 0]

a_mean = np.mean(a)
b_mean = np.mean(b)
c_mean = np.mean(c)

a_sd = np.std(a)
b_sd = np.std(b)
c_sd = np.std(c)

s = np.ndarray((4, 3))
for i in range(len(a)):
    s[i][0] = (a[i] - a_mean) / a_sd
    s[i][1] = (b[i] - b_mean) / b_sd
    s[i][2] = (c[i] - c_mean) / c_sd

s_t = s.T

print(s)
print(s_t)
cm = np.matmul(s_t, s) / 4

print(cm)
