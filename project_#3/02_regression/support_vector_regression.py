import datasets.dataset_provider as data_provider
import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """ Reading Data """
    return data_provider.get_position_salaries()


def visualise_linear_results(X, y, X_grid, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, predictions, color='blue')
    plt.title('Truth of Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # # Feature Scalling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y.reshape(-1, 1))

    regressor = SVR(kernel='rbf')
    regressor.fit(X, y)

    # Visualising the Regression results
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    visualise_linear_results(X, y, X_grid, regressor.predict(X_grid))

    # Predicting a new result with Regression
    predict = regressor.predict(sc_X.transform(np.array([[6.5]])))

    print('Predictions is %d' % sc_y.inverse_transform(predict))
