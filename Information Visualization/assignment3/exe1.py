import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read data
def read_data(file):
    data = pd.read_csv(file)
    return data


# (a), 3 dimensional scatter plot:

def t_scatter(data):
    x = range(0, 1000)
    # x = np.random.choice(x, 500)
    t = data['t'].values
    # t = np.take(t, x)
    plt.scatter(x, t, c=t, s=0.1, cmap='plasma', marker='x')
    plt.title('Scatter Plot of Vector t')
    plt.colorbar()
    plt.show()


# x = np.random.random(10)
# y = np.random.random(10)
#
# plt.scatter(x, y, c=y, s=500, cmap='plasma')
# plt.colorbar()
# plt.show()

data = read_data('./knotty_data.csv')
t_scatter(data)
