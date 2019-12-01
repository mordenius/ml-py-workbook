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


def initialises():
    shape = (2, 3)
    # All 0s matrix
    zeros_np = np.zeros(shape)

    # All 1s matrix
    np.ones(shape)

    # Any number
    np.full(shape, 99)

    # Shape from another
    np.full_like(zeros_np, 99)

    # Random decimal number
    np.random.rand(shape[0], shape[1])

    # Random int number
    np.random.randint(-5, 7, size=shape)

    # The identity matrix
    np.identity(5)

    # Repeat an array
    arr = [[1, 2, 3]]
    np.repeat(arr, 3, axis=1)

    # Creative
    output = np.ones((5, 5))

    z = np.zeros((3, 3))
    z[1, 1] = 9

    output[1:-1, 1:-1] = z
    print(output)

    # Copying
    a = np.array([1, 2, 3])
    b = a.copy()
    print(b)


def mathematics():
    a = np.array([1, 2, 3, 4])

    print(a + 2)
    print(a - 2)
    print(a * 2)
    print(a / 2)
    print(a ** 3)

    # Take the sin
    np.cos(a)

    # Reference https: // docs.scipy.org / doc / numpy / reference / routines.math.html


def linear_algebra():
    a = np.ones((2, 3))
    b = np.full((3, 2), 2)

    print(np.matmul(a, b))

    # Find the determinant
    c = np.identity(3)
    np.linalg.det(c)

    # Reference (https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)
    # Determinant
    # Trace
    # Singular Vector Decomposition
    # Eigenvalues
    # Matrix Norm
    # Inverse
    # etc...


def statistics():
    stats = np.array([[1, 2, 3], [4, 5, 6]])

    np.min(stats)  # 1
    np.max(stats)  # 6

    np.min(stats, axios=0)  # [3, 6]

    np.sum(stats)  # 21
    np.sum(stats, axis=1)  # [5, 7, 9]


if __name__ == '__main__':
    # basics()
    # accessing()
    # initialises()
    # mathematics()
    # linear_algebra()
    statistics()
