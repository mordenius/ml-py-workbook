import datasets.dataset_provider as data_provider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """ Reading Data """
    return data_provider.get_social_network_ads()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
    classifier.fit(X_train, y_train)

    # Predicting a new result with Regression
    predictions = classifier.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
