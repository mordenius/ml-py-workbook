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


def get_country_salaries():
    return _read_from_file('./project_#3/02_section_data.csv')


def get_experience_salaries():
    return _read_from_file('./project_#3/04_section_salary_data.csv')


def get_position_salaries():
    return _read_from_file('./project_#3/06_position_salaries.csv')


def get_startups():
    return _read_from_file('./project_#3/05_fifty_startups.csv')


def get_social_network_ads():
    return _read_from_file('./project_#3/12_social_network_ads.csv')


class Dataset(enum.Enum):
    iris = 1
    head_brain = 2
    titanic = 3
    cardio = 4
    country_salaries = 5
    experience_salaries = 6
    position_salaries = 7
    startups = 8
    social_network_ads = 9


_datasets = (None, get_iris, get_head_brain, get_titanic, get_cardio, get_country_salaries, get_experience_salaries,
             get_position_salaries, get_startups, get_social_network_ads)


def get_dataset(dataset):
    return _datasets[dataset]()
