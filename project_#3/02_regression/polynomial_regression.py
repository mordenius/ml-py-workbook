import datasets.dataset_provider as data_provider
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def load_dataset():
    """ Reading Data """
    return data_provider.get_position_salaries()


def visualise_linear_results(X, y, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X, predictions, color='blue')
    plt.title('Truth of Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


def visualise_polynomial_results(X, y, X_grid, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, predictions, color='blue')
    plt.title('Truth of Bluff (Polynomial Regression))')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Fitting Linear Regression to the dataset
    linear_regression = LinearRegression()
    linear_regression.fit(X, y)

    # Fitting Polynomial Regression to the dataset
    polynomial_regression = PolynomialFeatures(degree=2)  # try 5
    X_poly = polynomial_regression.fit_transform(X)
    polynomial_regression.fit(X_poly, y)

    linear_over_poly_regression = LinearRegression()
    linear_over_poly_regression.fit(X_poly, y)

    # Visualising the Linear Regression results
    # visualise_linear_results(X, y, linear_regression.predict(X))

    # Visualising the Polynomial Linear Regression results
    # X_grid = np.arange(min(X), max(X), 0.1)
    # X_grid = X_grid.reshape((len(X_grid), 1))
    # predictions = linear_over_poly_regression.predict(polynomial_regression.fit_transform(X_grid))
    #
    # visualise_polynomial_results(
    #     X, y, X_grid, predictions)

    # Predicting a new result with Linear Regression
    linear_predict = linear_regression.predict(np.array(6.5).reshape(-1,1))

    # Predicting a new result with Polymial Regression
    polynomial_predict = linear_over_poly_regression.predict(polynomial_regression.fit_transform(np.array(6.5).reshape(-1,1)))

    print('Predictions: linear is %ds and polynomial is %ds' % (linear_predict, polynomial_predict))
