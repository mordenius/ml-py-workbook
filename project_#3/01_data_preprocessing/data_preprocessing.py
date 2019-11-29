import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    ''' Reading Data '''

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/02_section_data.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


if __name__ == "__main__":
    dataset = load_dataset()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    missingvalues = SimpleImputer(
        missing_values=np.nan, strategy='mean', verbose=0)
    missingvalues = missingvalues.fit(X[:, 1:3])
    X[:, 1:3] = missingvalues.transform(X[:, 1:3])

    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X), dtype=np.float)

    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train,  y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
