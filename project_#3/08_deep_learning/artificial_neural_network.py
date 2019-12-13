import datasets.dataset_provider as data_provider
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_dataset():
    """ Reading Data """
    return data_provider.get_churn_modelling()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # Encoding categorical data
    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), [1, 2])], remainder='passthrough')
    X = np.array(ct.fit_transform(X), dtype=np.float)
