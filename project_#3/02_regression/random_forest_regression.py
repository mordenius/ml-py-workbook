import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    """ Reading Data """

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/06_position_salaries.csv')
    return pd.read_csv(full_path, sep=',')


def visualise_results(X, y, X_grid, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, predictions, color='blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, y)

    # Visualising the Regression results
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    visualise_results(X, y, X_grid, regressor.predict(X_grid))

    # Predicting a new result with Regression
    predict = regressor.predict(np.array([[6.5]]))

    print('Predictions is %d' % predict)