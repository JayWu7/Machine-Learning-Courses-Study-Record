import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_preprocess import preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from LinearRegression import multiple_regression


def pca_process(data):
    # Standardize the Data

    features = ['HTHG', 'HTAG', 'HST', 'AST', 'HS', 'AS']
    # Separating out the features
    x = data.loc[:, features].values
    # Separating out the target
    y = data.loc[:, ['FTG']].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', 'principal component 3',
                                          'principal component 4', 'principal component 5'])

    finalDf = pd.concat([principalDf, data[['FTG']]], axis=1)

    print(finalDf)

    plot_projection(finalDf)

    # plot Explained Variance

    plot_explained_variance(pca.explained_variance_ratio_)
    print(type(pca.explained_variance_ratio_))

    return finalDf


def plot_projection(df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = range(0, 7)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for target, color in zip(targets, colors):
        indicesToKeep = df['FTG'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1'], df.loc[indicesToKeep, 'principal component 2'],
                   c=color, s=30)
    ax.legend(targets)
    ax.grid()
    plt.show()


def plot_explained_variance(vars):
    plt.plot(list(range(1, len(vars) + 1)), vars.cumsum(), '-p', color='gray',
             markersize=15, linewidth=4,
             markerfacecolor='white',
             markeredgecolor='gray',
             markeredgewidth=2)
    my_x_ticks = np.arange(0, len(vars) + 1, 1)
    plt.xticks(my_x_ticks)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    data = preprocess()
    train_data = preprocess('test')
    pca_data = pca_process(data)
    pca_test_data = pca_process(train_data)
    features = pca_data.columns[:-1]

    # using pca data to do the regression job
    multiple_regression(pca_data, pca_test_data, features)
