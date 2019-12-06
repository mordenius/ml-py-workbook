import math
import random

import datasets.dataset_provider as data_provider
import matplotlib.pyplot as plt

dataset = data_provider.get_ads_ctr_optimisation()


class Selections:
    def __init__(self, algorithm):
        self.N = 10000
        self.d = 10
        self.algorithm = algorithm
        self.ads_selected = []
        self.total_reward = 0

    def start(self):
        for n in range(0, self.N):
            self.algorithm(self, n)


def random_algorithm(selections, n):
    ad = random.randrange(selections.d)
    selections.ads_selected.append(ad)
    reward = dataset.values[n, ad]
    selections.total_reward = selections.total_reward + reward


numbers_of_selections = [0] * 10
sums_of_rewards = [0] * 10


def upc_algorithm(selections, n):
    ad = 0
    max_upper_bound = 0
    for i in range(0, selections.d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    selections.ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    selections.total_reward += reward


random_selection = Selections(random_algorithm)
random_selection.start()

ucp_selection = Selections(upc_algorithm)
ucp_selection.start()

fig = plt.figure()
plt.subplots_adjust(left=0.08, bottom=0.16, right=0.95, wspace=0.2, hspace=0)

# Visualising the results
left = plt.subplot2grid((1, 2), (0, 0))
left.hist(random_selection.ads_selected)
plt.title('Histogram of ads selections (Random)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

# Visualising the results
right = plt.subplot2grid((1, 2), (0, 1))
right.hist(ucp_selection.ads_selected)
plt.title('Histogram of ads selections (UCB)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')

plt.show()
