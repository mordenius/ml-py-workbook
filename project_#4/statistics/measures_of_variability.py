import math

from .measures_of_central_tendency import get_mean, get_quantile


def get_data_range(sample):
    return max(sample) - min(sample)


def get_de_mean(sample):
    x_bar = get_mean(sample)
    return [x_i - x_bar for x_i in sample]


def get_variance(sample):
    length = len(sample)
    deviations = get_de_mean(sample)
    return get_sum_of_squares_total(deviations) / (length - 1)


def get_sum_of_squares_total(sample):
    mean = get_mean(sample)
    return sum(pow(element - mean, 2) for element in sample)


def get_sum_of_squares_residual(sample, predicted):
    return sum(pow(sample[index] - predicted[index], 2) for index in range(len(sample)))


def get_standard_deviation(sample):
    mean = get_mean(sample)
    summary = 0.
    for element in sample:
        summary += pow(float(element - mean), 2)
    D = summary / (len(sample) - 1)
    return math.sqrt(D)


def get_standard_deviation2(sample):
    return math.sqrt(get_variance(sample))


def get_interquartile_range(sample):
    return get_quantile(sample, 0.75) - get_quantile(sample, 0.25)
