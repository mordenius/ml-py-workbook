import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import statsmodels.regression.linear_model as sm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    """ Reading Data """

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/05_fifty_startups.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


if __name__ == "__main__":
    dataset = load_dataset()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X), dtype=np.float)

    y = LabelEncoder().fit_transform(y)

    # Avoiding the Dummy Variable trap
    X = X[:, 1:]

    # Building the optimal model using Backward Elimination
    X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
    X_opt = X[:, [0, 1, 2, 3, 4, 5]]

    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

    X_opt = X[:, [0, 1, 2, 3, 5]]

    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

    X_opt = X[:, [0, 2, 3, 5]]

    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

    X_opt = X[:, [0, 3, 5]]

    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

    X_opt = X[:, [0, 3]]

    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    print(regressor_OLS.summary())

    X_train, X_test, y_train, y_test = train_test_split(
        X_opt, y, test_size=0.2, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    predictions = regression_model.predict(X_test)
    r2_score = regression_model.score(X_opt, y)

    print(r2_score)

    # plt.scatter(X_train, y_train, color='red')
    # plt.scatter(X_test, y_test, color='green')
    # plt.plot(X_train, regression_model.predict(X_train), color='blue')
    # plt.show()
