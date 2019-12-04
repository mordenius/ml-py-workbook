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
    return _read_from_file('./datasets/head_brain.csv')


def get_titanic():
    return _read_from_file('./datasets/titanic.csv')


def get_pima():
    return _read_from_file('./datasets/pima_data_orig.csv')


def get_retail_online():
    return _read_from_file('./datasets/online_retail.csv')


def get_movie_metadata():
    return _read_from_file('./datasets/movie_metadata.csv')


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


def get_mall_customers():
    return _read_from_file('./project_#3/21_mall_customers.csv')


class Dataset(enum.Enum):
    iris = 1
    head_brain = 2
    titanic = 3
    pima = 4
    retail_online = 5
    movie_metadata = 6
    cardio = 7
    country_salaries = 8
    experience_salaries = 9
    position_salaries = 10
    startups = 11
    social_network_ads = 12
    mall_customers = 13


_datasets = (
    None, get_iris, get_head_brain, get_titanic, get_pima, get_retail_online, get_movie_metadata, get_cardio,
    get_country_salaries, get_experience_salaries,
    get_position_salaries, get_startups, get_social_network_ads, get_mall_customers)


def get_dataset(dataset):
    return _datasets[dataset]()
