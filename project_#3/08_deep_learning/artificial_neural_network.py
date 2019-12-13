import datasets.dataset_provider as data_provider
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    # Initializing the ANN
    classifier = Sequential()

    # Adding input and first hidden layers
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))

    # Adding second hidden layer
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))

    # Adding output layer
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))
