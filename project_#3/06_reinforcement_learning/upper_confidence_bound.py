import math

numbers_of_selections = [0] * 10
sums_of_rewards = [0] * 10


def upc_algorithm(selections, n):
    ad = 0
    max_upper_bound = 0
    for i in range(0, selections.COUNT_OF_SUGGESTIONS):
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
    reward = selections.dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    selections.total_reward += reward
