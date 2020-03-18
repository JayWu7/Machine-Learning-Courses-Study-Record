import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot():
    data = pd.read_csv('./wine.data')
    data = data.loc[:, ['Varieties', 'Alcohol', 'Malic acid', 'Ash', 'Color intensity', 'Proline']]
    varieties = data['Varieties']
    data.loc[:, 'Varieties'] = ['Variety {}'.format(v) for v in varieties]

    sns.pairplot(data, hue='Varieties')
    # plt.show()
    plt.savefig('exe_4.png')


plot()
