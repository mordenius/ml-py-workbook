from collections import Counter


def get_mean(sample):
    return sum(sample) / len(sample)


def get_median(sample):
    length = len(sample)
    sorted_ = sorted(sample)  # https://en.wikipedia.org/wiki/Quickselect
    midpoint = length // 2

    if length % 2 == 1:
        return sorted_[midpoint]
    return (sorted_[midpoint - 1] + sorted_[midpoint]) / 2


def get_quantile(sample, percent):
    index = int(percent * len(sample))
    return sorted(sample)[index]


def get_mode(sample):
    counts = Counter(sample)
    max_count = max(counts.values())

    return [x_i for x_i, count in counts.items() if count == max_count]
