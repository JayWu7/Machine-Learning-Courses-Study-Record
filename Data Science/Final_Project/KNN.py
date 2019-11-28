import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from data_preprocess import read_data, read_test_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def knn(features: list):
    X_train, y_train = read_data()
    X_test, y_test = read_test_data()

    X_train = X_train[features].values
    X_test = X_test[features].values

    y_train = y_train['Interest'].values
    y_test = y_test['Interest'].values

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)

    # make prediction
    y_pred = classifier.predict(X_test)

    print(pd.DataFrame({'Actual:': y_test, 'Prediction:': y_pred}).sample(10))

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    # plot error when k changes from 1 to 40:
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    plot_confusing_matrix(confusion_matrix(y_test, y_pred))

def plot_confusing_matrix(cm):
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
    plt.figure(figsize=(12, 12))
    plt.title('Confusing Matrix', y=1.05, size=15)
    sns.heatmap(cm, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, vmin=200)
    plt.show()



knn(['HTHG', 'HTAG', 'HST', 'AST', 'HS', 'AS'])
