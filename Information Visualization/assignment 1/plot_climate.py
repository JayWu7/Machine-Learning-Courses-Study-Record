import matplotlib.pyplot as plt
import pandas as pd


def read_data():
    '''
    :return: the climate data with some pre-processing
    '''
    with open('./climate.data', 'r') as f:
        data = pd.read_csv(f)
        return data.rename(columns={name: name[1:] for name in data.columns})


def plot_part_line():
    data = read_data()
    date = ['{}/{}'.format(year, month) for year, month in zip(data['Year'], data['Mo'])]
    data.insert(2, "Date", date)
    data = data.loc[200:300, ]
    plt.figure(figsize=(12, 10))
    data = data.drop(columns=['Year', 'Mo'])
    x = data.loc[::2, 'Date']
    plt.plot(x, data.loc[::2, 'Globe'], '-k', label='Globe')
    plt.plot(x, data.loc[::2, 'Land'], ':g', label='Land')
    plt.plot(x, data.loc[::2, 'Ocean'], ':b', label='Ocean')
    plt.xticks(x[::5], rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Average degree')
    plt.legend()
    plt.title('Average degree change from {} to {}'.format(x.iloc[0], x.iloc[-1]))
    plt.savefig('./question_a.png', dpi=1000)


def plot_year():
    data = read_data()
    data = data.iloc[1:-1, :]
    plt.figure(figsize=(12, 10))
    x = data.loc[::4, 'Year']
    plt.plot(x, data.loc[::4, 'Globe'], '-k', label='Globe')
    plt.plot(x, data.loc[::4, 'Land'], ':g', label='Land')
    plt.plot(x, data.loc[::4, 'Ocean'], ':b', label='Ocean')
    plt.xticks(x[::12], rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Average degree')
    plt.legend()
    plt.title('Average degree change from {} to {}'.format(x.iloc[0], x.iloc[-1]))
    plt.savefig('question_b.png', dpi=1000)


plot_year()
