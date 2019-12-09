import math


def combinations_without_repetition(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def permutations_without_repetition(n, k=None):
    if k is None:
        return math.factorial(n)
    return math.factorial(n) / math.factorial(n - k)
