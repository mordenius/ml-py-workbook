import numpy as np


def basics():
    a = np.array([1, 2., 3.02])
    b = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 0]])

    # Get Dimension
    print('Dimension of array \'a\' is {}'.format(a.ndim))
    print('Dimension of array \'b\' is {}'.format(b.ndim))

    # Get Shape
    print('Shape of array \'a\' is {}'.format(a.shape))
    print('Shape of array \'b\' is {}'.format(b.shape))

    # Get Type
    print('Type of values in array \'a\' is {}'.format(a.dtype))
    print('Type of values in array \'b\' is {}'.format(b.dtype))

    # Get item Size
    print('Size of values in array \'a\' is {}'.format(a.itemsize))
    print('Size of values in array \'b\' is {}'.format(b.itemsize))

    # Get Array Size
    print('Size of array \'a\' is {}'.format(a.nbytes))
    print('Size of array \'b\' is {}'.format(b.nbytes))

    # Get Count of Values
    print('Count of Values in array \'a\' is {}'.format(a.size))
    print('Count of Values in array \'b\' is {}'.format(b.size))


if __name__ == '__main__':
    basics()

