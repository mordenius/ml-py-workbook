import datasets.dataset_provider as data_provider
from apyori import apriori

dataset = data_provider.get_market_basket_optimisation()

transactions = []
# for i in range(0, dataset.shape[0]):
for i in range(0, 100):
    transactions.append(str(dataset.values[i, j]) for j in range(0, 20))

min_support = (3 * 7) / dataset.shape[0]
min_confidence = 0.2  # 20%
min_lift = 3

rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=2)

result = list(rules)
print(result)
