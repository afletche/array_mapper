import numpy as np

from mapped_array import MappedArray

def dot(a, b):
    new_array = MappedArray(object=b)

    if type(b) is np.ndarray:
        new_array.linear_map = a

    elif type(b) is MappedArray:
        if b.linear_map is None:
            new_array.linear_map = a
        else:
            new_array.linear_map = np.dot(a, b.linear_map)

        if b.offset_map is None:
            pass
        else:
            new_array.offset_map = np.dot(a, b.offset_map)


    new_shape = tuple(a.shape[:-1] + b.shape[1:])
    new_array.shape = new_shape

    new_array.evaluate()

    return new_array


def matmul(a, b):
    return dot(a,b)


def matvec(a, b):
    return dot(a,b)



if __name__ == "__main__":
    input = np.array([1, 2, 3])
    a = MappedArray([1, 2, 3])  # this would be am.MappedArray(...) or am.array(...)

    print('a', a)

    map = np.arange(9).reshape((3,3))

    b = dot(map, a)     # the dot would be am.dot
    print('b', b)
    print('numpy_check', np.dot(map, input))
    print('dotting a numpy array', dot(map, input))

    map2 = np.arange(12).reshape((4,3))
    c = dot(map2, a)
    print('c', c)
    print('c.shape', c.shape)
    print('c_numpy_check', np.dot(map2, input))

    d = dot(map2, b)
    print('d', d)
    print('d_numpy_check', map2.dot(map).dot(input))

    input2 = np.array([2, 4, 6])
    print('d2', d.evaluate(input2))
    print('d2_numpy_check', map2.dot(map).dot(input2))
