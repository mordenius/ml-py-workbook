import os
import enum
import pandas

current_dir = os.path.dirname(os.path.realpath(__file__))


def _read_from_file(filename):
    return pandas.read_csv(os.path.join(current_dir, filename))


iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]


def getIris():
    return pandas.read_csv(iris_url, names=names)


def getHeadBrain():
    return _read_from_file('headbrain.csv')


def getTitanic():
    return _read_from_file('titanic.csv')


class Dataset(enum.Enum):
    iris = 1
    headbrain = 2
    titanic = 3


_datasets = (None, getIris, getHeadBrain, getTitanic)


def getDataset(dataset):
    return _datasets[dataset]()
