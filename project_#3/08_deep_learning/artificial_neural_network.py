import datasets.dataset_provider as data_provider


def load_dataset():
    """ Reading Data """
    return data_provider.get_churn_modelling()


if __name__ == '__main__':
    dataset = load_dataset()

    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
