import datasets.dataset_provider as data_provider
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans


def main():
    sns.set()  # for plot styling
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')


def load_dataset():
    dataset = data_provider.get_movie_metadata()
    return dataset


def is_numeric(value):
    """ Test if a value is numerid"""
    return isinstance(value, int) or isinstance(value, float)


if __name__ == "__main__":
    main()
    dataset = load_dataset()

    newdata = dataset.iloc[:, 4:6]
    # print(newdata.describe())

    newdata.dropna(axis=0, inplace=True)

    print("No. of columns containing null values")
    print(len(newdata.columns[newdata.isna().any()]))

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(newdata)

    print(kmeans.cluster_centers_)

    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    print(dict(zip(unique, counts)))

    newdata['cluster'] = kmeans.labels_
    sns.set_style('whitegrid')
    sns.lmplot('director_facebook_likes', 'actor_3_facebook_likes',
               data=newdata, hue='cluster', aspect=1, fit_reg=False)
    plt.show()
