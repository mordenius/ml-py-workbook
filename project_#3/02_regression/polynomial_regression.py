import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    """ Reading Data """

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/06_position_salaries.csv')
    return pd.read_csv(full_path, sep=',')


def visualise_linear_results(X, y, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X, predictions, color='blue')
    plt.title('Truth of Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


def visualise_polynomial_results(X, y, predictions):
    plt.scatter(X, y, color='red')
    plt.plot(X, predictions, color='blue')
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
    polynomial_regression = PolynomialFeatures(degree=2) # try 5
    X_poly = polynomial_regression.fit_transform(X)
    polynomial_regression.fit(X_poly, y)

    linear_over_poly_regression = LinearRegression()
    linear_over_poly_regression.fit(X_poly, y)

    # Visualising the Linear Regression results
    # visualise_linear_results(X, y, linear_regression.predict(X))

    # Visualising the Polynomial Linear Regression results
    predictions = linear_over_poly_regression.predict(polynomial_regression.fit_transform(X))
    visualise_polynomial_results(
        X, y, predictions)
