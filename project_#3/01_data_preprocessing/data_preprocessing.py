import os
import pandas as pd

from sklearn.preprocessing import Imputer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    ''' Reading Data '''

    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/02_section_data.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


def care_out_missing_data(dataset):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imputer.fit(dataset)


if __name__ == "__main__":
    dataset = load_dataset()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    imputer = care_out_missing_data(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])

    print(X)
