import matplotlib.pyplot as plt
from data_preprocess import preprocess
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np


def plot_two_features(data, f1, f2):
    data.plot(x=f1, y=f2, style='.', color='k')
    print(data[f1])
    print(data[f2])
    plt.title('{} vs {}'.format(f1, f2))
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()


def plot_density_feature(data, f):
    plt.figure(figsize=(18, 12))
    plt.tight_layout()
    seabornInstance.distplot(data[f])
    # plt.show()
    plt.savefig('{}.png'.format(f))


def simple_regression(X, y, X_test, y_test):
    '''
    :param X:  The attributes we used to predict the label y.
    :return:
    '''
    regressor = LinearRegression()
    regressor.fit(X, y)  # training the algorithm
    # To retrieve the intercept:
    print(regressor.intercept_)
    # For retrieving the slope:
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df1 = df.sample(30)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig('simple_regression.png')
    plt.show()
    plot_test_data(X_test, y_test, y_pred)
    calculate_error(y_test, y_pred)


def plot_test_data(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.savefig('regression_test.png')


def calculate_error(y_test, y_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def multiple_regression(data: pd.DataFrame, test_data: pd.DataFrame, features: list):
    # divide the data into “attributes” and “labels”

    X_train = data[features].values
    y_train = data['FTG'].values

    X_test = test_data[features].values
    y_test = test_data['FTG'].values

    # train our model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, index=features, columns=['Coefficient'])
    print(coeff_df)

    # do prediction
    y_pred = regressor.predict(X_test)

    # check the difference
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.sample(30)

    print(df1)

    # plot the difference
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig('multi_regression.png')
    plt.show()

    calculate_error(y_test, y_pred)
    # plot_test_data(X_test, y_test, y_pred)


def simple_model(fea):
    train_data = preprocess()
    test_data = preprocess('test')
    X = train_data[fea].values.reshape(-1, 1)
    y = train_data['FTG'].values.reshape(-1, 1)

    X_test = test_data[fea].values.reshape(-1, 1)
    y_test = test_data['FTG'].values.reshape(-1, 1)

    simple_regression(X, y, X_test, y_test)


def multiple_model(features):
    train_data = preprocess()
    test_data = preprocess('test')

    multiple_regression(train_data, test_data, features)
    plot_density_feature(train_data, 'FTG')


# plot_two_features(preprocess(), 'AST', 'FTG')

# simple_model('HTHG')
multiple_model(['HTHG', 'HTAG', 'HST', 'AST', 'HS', 'AS'])
