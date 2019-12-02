import math


def get_mean(sample):
    return sum(sample) / len(sample)


def get_standard_deviation(sample):
    mean = get_mean(sample)
    summary = 0.
    for element in sample:
        summary += pow(float(element - mean), 2)
    D = summary / (len(sample) - 1)
    return math.sqrt(D)


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


if __name__ == '__main__':
    test_sample = [1, 2, 2, 3, 4, 4, 5]
    test_predicted = [1, 2, 2, 3, 4, 3, 5]

    sd = get_standard_deviation(test_sample)
    r2 = get_r_of_squares(test_sample, test_predicted)
    adj_r2 = get_adjusted_r_of_squares(test_sample, test_predicted, 3)
