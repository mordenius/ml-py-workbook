import os
import math

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print('\nLogistic Regression Model Accuracy')


def resolveAccuracy(X_train, X_test, y_train, y_test):
    # Fitting
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


dir_path = os.path.dirname(os.path.realpath(__file__))

# Reading Data
dataset = pd.read_csv(os.path.join(
    dir_path, '../../datasets/datasets/social_network_ads.csv'), sep=',')

# Prepare Data
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


print('    Before Scaling: {}'.format(
    resolveAccuracy(X_train, X_test, y_train, y_test)))


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print('    After Scaling: {}'.format(
    resolveAccuracy(X_train, X_test, y_train, y_test)))
