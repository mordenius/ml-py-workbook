import datasets.dataset_provider as data_provider

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split


def load_dataset():
    dataset = data_provider.get_pima()
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
