import math
import datasets.dataset_provider as data_provider
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.rcParams['figure.figsize'] = (20., 10.)


# Reading Data
data = data_provider.get_titanic()

# Collecting X and Y
# X = data['Head Size(cm^3)'].values
# Y = data['Brain Weight(grams)'].values

# sns.countplot(x='Survived', hue='Sex', data=data)
# plt.show()

# sns.countplot(x='Survived', hue='Pclass', data=data)
# plt.show()

# data['Age'].plot.hist()
# plt.show()

# data['Fare'].plot.hist(pin=20, figsize=(10, 5))
# plt.show()

# Data Wrangling

# data.isnull().sum()
# sns.heatmap(data.isnull(), yticklabels=False)
# plt.show()

# sns.boxplot(x='Pclass', y='Age', data=data)
# plt.show()

data.drop('Cabin', axis=1, inplace=True)
data.dropna(inplace=True)
# sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
# plt.show()

# Binary Mask - Male True or False
sex = pd.get_dummies(data['Sex'], drop_first=True)
embark = pd.get_dummies(data['Embarked'], drop_first=True)
pclass = pd.get_dummies(data['Pclass'], drop_first=True)

prepared_data = pd.concat([data, sex, embark, pclass], axis=1)

prepared_data.drop(['Sex', 'Embarked', 'Pclass', 'PassengerId',
                    'Name', 'Ticket'], axis=1, inplace=True)


# Train Data
X = prepared_data.drop('Survived', axis=1)
y = prepared_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Create Logistic Model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# Predictions
predictions = logistic.predict(X_test)

class_results = classification_report(y_test, predictions)
print(class_results)

matrix_result = confusion_matrix(y_test, predictions)
print(matrix_result)

acc_score = accuracy_score(y_test, predictions)
print(acc_score)
