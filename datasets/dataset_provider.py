import os
import enum
import pandas

current_dir = os.path.dirname(os.path.realpath(__file__))


def _read_from_file(filename, sep=','):
    return pandas.read_csv(os.path.join(current_dir, filename), sep=sep)


iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]


def get_iris():
    return pandas.read_csv(iris_url, names=names)


def get_head_brain():
    return _read_from_file('head_brain.csv')


def get_titanic():
    return _read_from_file('titanic.csv')


def get_cardio():
    return _read_from_file('./project_#2/cardio_train.csv', sep=';')


class Dataset(enum.Enum):
    iris = 1
    head_brain = 2
    titanic = 3
    cardio = 4


_datasets = (None, get_iris, get_head_brain, get_titanic, get_cardio)


def get_dataset(dataset):
    return _datasets[dataset]()
