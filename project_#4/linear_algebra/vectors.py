import math


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
