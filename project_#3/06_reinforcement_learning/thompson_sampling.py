import random

number_of_rewards_1 = [0] * 10
number_of_rewards_0 = [0] * 10


def thompson_sampling(selections, n):
    ad = 0
    max_random = 0
    for i in range(0, selections.COUNT_OF_SUGGESTIONS):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    selections.ads_selected.append(ad)
    reward = selections.dataset.values[n, ad]
    selections.total_reward += reward
    number_of_rewards = number_of_rewards_1 if reward == 1 else number_of_rewards_0
    number_of_rewards[ad] += 1
