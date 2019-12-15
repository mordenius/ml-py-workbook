import datasets.dataset_provider as data_provider
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,

from xgboost import XGBClassifier

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

    # Fitting XGBoost to the Training set
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    # Predicting the Test set result
    predictions = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    print(accuracies.mean(), accuracies.std())
