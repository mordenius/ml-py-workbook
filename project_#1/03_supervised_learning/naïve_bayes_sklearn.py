import os
import pandas as pd

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    full_path = os.path.join(
        dir_path, './../../datasets/datasets/pima_data_orig.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset


if __name__ == '__main__':
    dataset = load_dataset()
    model = GaussianNB()

    X = dataset.drop('diabetes', axis=1)
    y = dataset['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc_score = metrics.accuracy_score(y_test, predictions)

    class_results = metrics.classification_report(y_test, predictions)
    print(class_results)

    matrix_result = metrics.confusion_matrix(y_test, predictions)
    print(matrix_result)

    print('Accuracy: {0}%'.format(acc_score * 100))
