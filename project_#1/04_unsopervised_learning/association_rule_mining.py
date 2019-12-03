import datasets.dataset_provider as data_provider

from mlxtend.frequent_patterns import apriori, association_rules


def load_dataset():
    dataset = data_provider.get_retail_online()
    return dataset


def encode_units(unit):
    if unit <= 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    dataset = load_dataset()
    # print(dataset.describe())

    # Clearing
    dataset['Description'] = dataset['Description'].str.strip()
    dataset.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    dataset['InvoiceNo'] = dataset['InvoiceNo'].astype('str')
    dataset = dataset[~dataset['InvoiceNo'].str.contains('C')]

    # print(dataset.head)

    basket = (dataset[dataset['Country'] == 'France'].groupby(['InvoiceNo', 'Description'])[
              'Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

    basket_sets = basket.applymap(encode_units)
    basket_sets.drop('POSTAGE', inplace=True, axis=1)

    frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    # print(rules.head())

    print(rules[(rules['lift'] >= 0) & (rules['confidence'] >= 0.8)])
