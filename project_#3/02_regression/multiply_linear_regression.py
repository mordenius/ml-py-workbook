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
    ''' Reading Data '''

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/05_fifty_startups.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


def backward_elimination(x, sl):
    num_vars = len(x[0])
    for i in range(0, num_vars):
        regressor_OLS = sm.OLS(y, x).fit()
        max_var = max(regressor_OLS.pvalues).astype(float)
        if max_var > sl:
            for j in range(0, num_vars - i):
                if regressor_OLS.pvalues[j].astype(float) == max_var:
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


def backward_elimination_and_adjusted_r_squared(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


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

    SL = 0.05
    X_opt = X[:, [0, 1, 2, 3, 4, 5]]

    X_Modeled = backward_elimination_and_adjusted_r_squared(X_opt, SL)
    print(X_Modeled)

    X_Modeled = backward_elimination(X_opt, SL)
    print(X_Modeled)

    X_train, X_test, y_train,  y_test = train_test_split(
        X_Modeled, y, test_size=0.2, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    predictions = regression_model.predict(X_test)
    r2_score = regression_model.score(X_Modeled, y)

    print(r2_score)

    # plt.scatter(X_train, y_train, color='red')
    # plt.scatter(X_test, y_test, color='green')
    # plt.plot(X_train, regression_model.predict(X_train), color='blue')
    # plt.show()
