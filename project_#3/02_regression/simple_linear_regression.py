import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    ''' Reading Data '''

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/04_section_salary_data.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


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
