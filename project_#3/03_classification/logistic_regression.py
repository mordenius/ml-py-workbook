import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    """ Reading Data """

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/12_social_network_ads.csv')
    return pd.read_csv(full_path, sep=',')


def show_plot_test_data(X, y):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('yellow', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.xlim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title('Logistic Regression')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0, solver='lbfgs')
    classifier.fit(X_train, y_train)

    # Predicting a new result with Regression
    predictions = classifier.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    show_plot_test_data(X_test, y_test)
