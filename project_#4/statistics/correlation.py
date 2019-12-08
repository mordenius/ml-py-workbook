from .. import vectors
from .measures_of_variability import get_standard_deviation, get_de_mean
from .measures_of_central_tendency import get_mean


def get_covariance(sample, predicted):
    length = len(sample)
    return vectors.dot(get_de_mean(sample), get_de_mean(predicted)) / (length - 1)


def get_correlation(sample, predicted):
    stdev_sample = get_standard_deviation(sample)
    stdev_predicted = get_standard_deviation(predicted)
    if (stdev_sample > 0 and stdev_predicted > 0):
        return get_covariance(sample, predicted)
    return 0  # if variables don't change, correlation is zero


def get_sum_of_squares_total(sample):
    mean = get_mean(sample)
    return sum(pow(element - mean, 2) for element in sample)


def get_sum_of_squares_residual(sample, predicted):
    return sum(pow(sample[index] - predicted[index], 2) for index in range(len(sample)))


def get_r_of_squares(sample, predicted):
    ssr = get_sum_of_squares_residual(sample, predicted)
    sst = get_sum_of_squares_total(sample)
    return 1 - (ssr / sst)


def get_adjusted_r_of_squares(sample, predicted, count_of_regressors):
    r_of_squares = get_r_of_squares(sample, predicted)
    return 1 - (1 - r_of_squares) / ((len(sample) - 1) / (len(sample) - count_of_regressors - 1))
