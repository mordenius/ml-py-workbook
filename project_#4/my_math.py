import math
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


def get_data_range(sample):
    return max(sample) - min(sample)


def get_de_mean(sample):
    x_bar = get_mean(sample)
    return [x_i - x_bar for x_i in sample]


def get_variance(sample):
    length = len(sample)
    deviations = get_de_mean(sample)
    return get_sum_of_squares_total(deviations) / (length - 1)


def get_standard_deviation(sample):
    mean = get_mean(sample)
    summary = 0.
    for element in sample:
        summary += pow(float(element - mean), 2)
    D = summary / (len(sample) - 1)
    return math.sqrt(D)


def get_interquartile_range(sample):
    return get_quantile(sample, 0.75) - get_quantile(sample, 0.25)


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


def vector_add(v, w):
    '''Vectors addition by element'''
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_substract(v, w):
    '''Vectors subtract by element'''
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    '''Vectors addition by element'''
    # recuce(vector_add, vectors)
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result


def scalar_multiply(scalar, vector):
    return [scalar * v_i for v_i in vector]


def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v, w):
    '''v_i * w_i + ... + v_n * w_n'''
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    '''v_i * v_i + ... + v_n * v_n'''
    return dot(v, v)


def magnitude(v):
    return math.sqrt(sum_of_squares())


def squared_distance(v, w):
    '''Squared distance between two vectors

    (v_i - w_i) ** 2 + ... + (v_n * w_n) ** 2'''
    return sum_of_squares(vector_substract(v, w))


def distance(v, w):
    '''Distance between two vectors'''
    return magnitude(vector_substract(v, w))


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A, i):
    return A[i]


def get_column(A, j):
    return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]


def is_diagonal(i, j):
    return 1 if i == j else 0


if __name__ == '__main__':
    test_sample = [1, 2, 2, 3, 4, 4, 5]
    test_predicted = [1, 2, 2, 3, 4, 3, 5]

    get_quantile(test_sample, 0.25)
    get_quantile(test_sample, 0.75)
    get_quantile(test_sample, 0.90)

    sd = get_standard_deviation(test_sample)
    r2 = get_r_of_squares(test_sample, test_predicted)
    adj_r2 = get_adjusted_r_of_squares(test_sample, test_predicted, 3)

    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4], [5, 6]]

    identify_matrix = make_matrix(5, 5, is_diagonal)
