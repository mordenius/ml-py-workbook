import random

import datasets.dataset_provider as data_provider
import matplotlib.pyplot as plt

dataset = data_provider.get_ads_ctr_optimisation()

from upper_confidence_bound import upc_algorithm


class Selections:
    def __init__(self, algorithm, dataset):
        self.algorithm = algorithm
        self.ads_selected = []
        self.total_reward = 0

        self.dataset = dataset
        self.COUNT_OF_SELECTIONS, self.COUNT_OF_SUGGESTIONS = dataset.shape

    def start(self):
        for n in range(0, self.COUNT_OF_SELECTIONS):
            self.algorithm(self, n)


def random_algorithm(selections, n):
    ad = random.randrange(selections.COUNT_OF_SUGGESTIONS)
    selections.ads_selected.append(ad)
    reward = dataset.values[n, ad]
    selections.total_reward = selections.total_reward + reward


random_selection = Selections(random_algorithm, dataset)
random_selection.start()

ucp_selection = Selections(upc_algorithm, dataset)
ucp_selection.start()

fig = plt.figure()
plt.subplots_adjust(left=0.08, bottom=0.16, right=0.95, wspace=0.2, hspace=0.5)

# Visualising the results
first = plt.subplot2grid((2, 1), (0, 0))
first.hist(random_selection.ads_selected)
plt.title('Histogram of ads selections (Random)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

# Visualising the results
second = plt.subplot2grid((2, 1), (1, 0))
second.hist(ucp_selection.ads_selected)
plt.title('Histogram of ads selections (UCB)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

plt.show()
