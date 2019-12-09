import math


def combinations_without_repetition(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
