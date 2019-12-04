import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_data():
    df_train_x = pd.read_csv('Football_data/football_train_x.csv')
    df_train_y = pd.read_csv('Football_data/football_train_y.csv')
    return df_train_x, df_train_y


def read_test_data():
    df_test_x = pd.read_csv('Football_data/football_test_x.csv')
    df_test_y = pd.read_csv('Football_data/football_test_y.csv')
    return df_test_x, df_test_y


def correlation(features):
    '''
    :param features: pd.dataframe
    :return: correlation matrix
    '''
    cor_matrix = features.corr(method="pearson")
    return cor_matrix


def plot_correlation(cor_matrix):
    '''
    :param cor_matrix: correlation matrix
    :return: plot the correlation matrix
    '''
    # f = plt.figure(figsize=(20, 20))
    # plt.matshow(cor_matrix.corr(), fignum=f.number)
    # plt.xticks(range(cor_matrix.shape[1]), cor_matrix.columns, fontsize=16, rotation=45)
    # plt.yticks(range(cor_matrix.shape[1]), cor_matrix.columns, fontsize=16)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=16)
    # plt.title('Correlation Matrix', fontsize=40)
    # plt.show()
    colormap = plt.cm.viridis
    plt.figure(figsize=(20, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(cor_matrix, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    plt.savefig('correlation.png')


def plot_pairs(df, data):
    pairs = []
    for i in df.index:
        for c in df.columns:
            if 0.4 <= abs(df.loc[i, c]) < 1 and (c, i) not in pairs:
                pairs.append((i, c))
    for pair in pairs:
        sns.pairplot(data, height=3, vars=pair, kind='reg', markers="+", diag_kind="kde")
        plt.savefig('{}_{}.png'.format(pair[0], pair[1]))


def plot_feature(data):
    plt.hist(data, 10, edgecolor='grey')
    plt.show()


def preprocess(flag='train'):
    if flag == 'train':
        x, y = read_data()
    elif flag == 'test':
        x, y = read_test_data()
    else:
        raise ValueError

    x['Interest'] = y['Interest']
    x['FTG'] = y['FTG']
    # cor_matrix = correlation(x)
    # plot_correlation(cor_matrix)

    # plot_pairs(cor_matrix, x)
    plot_feature(x['FTG'].values)
    return x


if __name__ == '__main__':
    preprocess()
