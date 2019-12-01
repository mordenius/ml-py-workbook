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


def accessing():
    a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]], dtype='int8')

    # Get a specific element
    # [row, column]
    print('Item on second row and fifth column is {}'.format(a[1, 4]))

    # Get a specific row
    print('First row is {}'.format(a[0, :]))

    # Get a specific column
    print('Third column is {}'.format(a[:, 2]))

    # Slice with sized step
    print('Even from second and one before end from first row is {}'.format(a[0, 1:-1:2]))

    # Change by index
    print('Before value is {}'.format(a[0, :]))
    a[0, 3] = 25
    print('After value is {}'.format(a[0, :]))

    # Change by index with array
    print('Before value is {}'.format(a[:, 4]))
    a[:, 4] = [96, 73]
    print('After value is {}'.format(a[:, 4]))

    # 3d
    b = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    # Get specific element
    print('--> {}'.format(b[0, 1, 1]))

    # replace
    b[:, 1, :] = [[7, 7, 7], [8, 8, 8]]
    print('--> {}'.format(b))


if __name__ == '__main__':
    # basics()
    accessing()
