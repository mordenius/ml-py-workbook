import datasets.dataset_provider as data_provider

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_dataset():
    """ Reading Data """
    return data_provider.get_position_salaries()


if __name__ == "__main__":
    dataset = load_dataset()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train,  y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    predictions = regression_model.predict(X_test)
    r2_score = regression_model.score(X, y)

    print(r2_score)

    plt.scatter(X_train, y_train, color='red')
    plt.scatter(X_test, y_test, color='green')
    plt.plot(X_train, regression_model.predict(X_train), color='blue')
    plt.show()
