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


if __name__ == '__main__':
    test_sample = [1, 2, 2, 3, 4, 4, 5]

    sd = get_standard_deviation(test_sample)
