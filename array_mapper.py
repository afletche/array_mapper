import numpy as np
import scipy.sparse as sps

def array(input=None, linear_map=None, offset=None, shape=None):
    '''
    Creates a MappedArray.

    Parameters
    ----------
    object: array_like
        The value of the MappedArray

    shape: tuple
        The shape of the MappedArray
    '''
    return MappedArray(input=input, linear_map=linear_map, offset=offset, shape=shape)


class MappedArray:
    ''' Sketch of MappedArray class for purpose of developing sample run script '''
    
    def __init__(self, input=None, linear_map=None, offset=None, shape=None) -> None:
        '''
        Creates an instance of a MappedArray object.

        Parameters
        ----------
        input: array_like
            The value of the MappedArray

        shape: tuple
            The shape of the MappedArray
        '''
        
        # Listing list of attributes
        self.input = input
        self.linear_map = linear_map
        self.offset_map = offset
        self.shape = shape
        self.value = None

        if type(input) is np.ndarray:
            self.input = input
            self.value = input
            self.shape = input.shape
        elif type(input) is list or type(input) is tuple:
            self.input = np.array(input)
            self.value = np.array(input)
            self.shape = np.array(input).shape
        elif type(input) is MappedArray and linear_map is not None:
            raise Exception("Can't instantiate the input with a MappedArray while specifying a linear map."
            "Please use the array_mapper.dot function.")
        elif type(input) is MappedArray:   # Creates a copy of the input MappedArray
            self.linear_map = input.linear_map
            self.offset_map = input.offset_map
            self.shape = input.shape
            self.input = input.input
            self.value = input.value

        if self.linear_map is None and self.input is not None:
            self.linear_map = sps.eye(self.input.shape[0])

        if type(shape) is list or type(shape) is tuple:
            self.shape = shape

        self.evaluate()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'array_mapper.MappedArray(input={self.input}, linear_map={self.linear_map}, offset={self.offset_map}, shape={self.shape})'

    def __pos__(self):
        return self

    def __neg__(self):
        map = None
        offset = None
        input = self.input
        if self.linear_map is not None:
            map = -self.linear_map
        else:
            input = -self.input

        if self.offset_map is not None:
            offset = -self.offset_map

        return array(input=input, linear_map=map, offset=offset, shape=self.shape)

    def __add__(self, x2):
        return add(self, x2)

    def __radd__(self, x1):
        return add(self, x1)

    def __sub__(self, x2):
        return add(self, -x2)

    def __rsub__(self, x1):
        return add(-self, x1)

    def __mul__(self, alpha):
        map = None
        offset = None
        if self.linear_map is not None:
            map = alpha*self.linear_map
        if self.offset_map is not None:
            offset = alpha*self.offset_map

        return array(input=self.input, linear_map=map, offset=offset, shape=self.shape)

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def __truediv__(self, alpha):
        return self.__mul__(1/alpha)

    def reshape(self, newshape):
        new_array = MappedArray(input=self.input, linear_map=self.linear_map, offset=self.offset_map, shape=newshape)
        return new_array

    def evaluate(self, input=None):
        if input is not None:
            if type(input) is MappedArray:
                new_array = dot(self.linear_map, input, offset=self.offset_map)
                return new_array.value
            else:
                self.input = input
        if self.linear_map is not None and self.offset_map is not None:
            self.value = self.linear_map.dot(self.input) + self.offset_map
        elif self.linear_map is not None:
            self.value = self.linear_map.dot(self.input)
        elif self.offset_map is not None:
            self.value = self.input + self.offset_map
        else:
            self.value = self.input

        self.value = self.value.reshape(self.shape)

        return self.value


def add(x1, x2, combine_input:bool=None):
    '''
    Adds the two arguments.

    Parameters
    -----------
    x1: array_like
        The first argument being added
    x2: array_like
        The second argument being added
    combine_input: bool
        If both arguments are MappedArrays, this boolean determines whether
        the output MappedArray should stack the inputs of x1 and x2 as its
        input (False), or whether the output MappedArray should use the same
        input (True). None option automatically detects if arguments share
        same input. If so, inputs are combined.
    '''
    if type(x1) is not MappedArray and type(x2) is not MappedArray:
        return x1 + x2
    elif type(x1) is not MappedArray:
        return add(x2=x2, x1=x1, combine_input=combine_input)

    map = None
    offset_map = None
    input = None

    if type(x2) is MappedArray:
        combine_input = _check_whether_to_combine_inputs(x1, x2)

        # if x1.linear_map is not None and x2.linear_map is not None:
        if combine_input:
            map = x1.linear_map + x2.linear_map
        else:
            if type(x1.linear_map) is np.ndarray or x2.linear_map is np.ndarray:
                map = np.hstack((x1.linear_map, x2.linear_map))
            else:
                map = sps.hstack((x1.linear_map, x2.linear_map))
                map = map.tocsc()

        if x1.offset_map is not None and x2.offset_map is not None:
            offset_map = x1.offset_map + x2.offset_map
        elif x1.offset_map is not None:
            offset_map = x1.offset_map
        elif x2.offset_map is not None:
            offset_map = x2.offset_map

        if combine_input:
            input = x1.input
        else:
            if len(x1.input.shape) == 1:
                input = np.append(x1.input, x2.input)
            else:
                input = np.vstack((x1.input, x2.input))

    elif type(x2) is np.ndarray or type(x2) is int or type(x2) is float:
        map = x1.linear_map
        input = x1.input
        if x1.offset_map is not None:
            offset_map = x1.offset_map + x2
        else:
            offset_map = x2

    new_array = array(input=input, linear_map=map, offset=offset_map, shape=x1.shape)

    return new_array


def subtract(x1, x2, combine_inputs:bool=None):
    '''
    Adds the two arguments.

    Parameters
    -----------
    x1: array_like
        The first argument being added
    x2: array_like
        The second argument being added
    combine_input: bool
        If both arguments are MappedArrays, this boolean determines whether
        the output MappedArray should stack the inputs of x1 and x2 as its
        input (False), or whether the output MappedArray should use the same
        input (True). None option automatically detects if arguments share
        same input. If so, inputs are combined.
    '''
    return add(x1, -x2, combine_input=combine_input)


def dot(map, input, offset=None):
    '''
    Dot product of a map and an input array.

    Parameters
    ----------
    map: numpy.ndarray or scipy.sparse._csc.csc_matrix
        The map that the input will be passed through. Must have a dot method.

    input: array_like
        The input array.
    '''
    new_array = MappedArray(input=input)

    if type(input) is np.ndarray or type(input) is sps._csc.csc_matrix:
        new_array.linear_map = map
    elif type(input) is MappedArray:
        if input.linear_map is None:
            new_array.linear_map = map
        else:
            if type(map) is not np.ndarray and type(input.linear_map) is not np.ndarray \
                or type(map) is np.ndarray and type(input.linear_map) is np.ndarray:    # if both sparse or both dense, just multiply
                new_array.linear_map = map.dot(input.linear_map)
            else:   # make sure the input is dense so the types work out in dot (just can't have a dense map and sparse input)
                input_linear_map = input.linear_map.copy()
                if type(input.linear_map) is not np.ndarray:
                    input_linear_map = input_linear_map.toarray()
                new_array.linear_map = map.dot(input_linear_map)

        input_offset_map = input.offset_map
        if type(input.offset_map) is int or type(input.offset_map) is float:
            input_offset_map = np.ones(input.shape)*input.offset_map

        if offset is not None and offset is not None:
            new_array.offset_map = map.dot(input_offset_map) + offset
        elif input.offset_map is not None:
            new_array.offset_map = map.dot(input_offset_map)
        elif offset is not None:
            new_array.offset_map = offset
        #Else new_array.offset_map = None
    else:
        new_array.linear_map = map


    new_shape = tuple(map.shape[:-1] + input.shape[1:])
    new_array.shape = new_shape

    new_array.evaluate()

    return new_array


def matmul(map, input):
    '''
    Matrix-matrix product between a map and an input matrix.

    Parameters
    ----------
    map: numpy.ndarray
        The map that the input will be passed through.

    input: array_like
        The input array.
    '''
    return dot(map,input)


def matvec(map, input):
    '''
    Matrix-vector product between a map and an input vector.

    Parameters
    ----------
    map: numpy.ndarray
        The map that the input will be passed through.

    input: array_like
        The input array.
    '''
    return dot(map,input)



def linear_combination(start, stop, num_steps=50, start_weights=None, stop_weights=None, combine_input=None, offset=None):
    '''
    Perform a linear combintation between two arrays.
    The input is the input of the start array vertically stacked with the input
    of the stop array unless if the inputs are combined. The inputs are
    combined if combine_input=True or if combine_input=None and the start and
    stop have the same input.

    Parameters
    ----------
    start: array_like
        The first array that will be used in the combination.

    stop: array_like
        The second array.

    num_steps: int
        The number of steps in the combination.

    start_weights: array_like
        The weights on the start input for the linear combination.

    stop_weights: array_like
        The weights on the stop input for the linear combination.

    combine_input : boolean
        A boolean on whether the two inputs share their inputs. If so, the inputs are merged for this array.

    offset: np.ndarray
        A constant offset applied after the linear evaluation.
    '''
    #dims_dont_match = pointset_start.shape != pointset_end.shape
    # if any(dims_dont_match): #pset1 and pset2 must have same number of points
    #     print('The sets you are trying to interpolate do not have the same dimensions.\n')
    #     return

    # TODO Add checking to make sure function inputs are correct sizes.
    # TODO They can't specify num_steps and weights.

    # if they specify num_steps and not weights, call linspace which will call this func with even weights.
    if num_steps is not None and start_weights is None and stop_weights is None:
        linspace(start, stop, num_steps, combine_input, offset)

    # num_per_step = start.value.shape[0]
    num_per_step = _num_elements(start.value)
    map_num_outputs = num_steps*num_per_step
    map_num_inputs = num_per_step
    map_start = sps.lil_array((map_num_outputs, map_num_inputs))
    map_stop = sps.lil_array((map_num_outputs, map_num_inputs))
    for i in range(num_steps):
        start_step_map = (sps.eye(num_per_step).tocsc()) * start_weights[i]
        map_start[i*num_per_step:(i+1)*num_per_step, :] = start_step_map

        stop_step_map = (sps.eye(num_per_step).tocsc()) * stop_weights[i]
        map_stop[i*num_per_step:(i+1)*num_per_step, :] = stop_step_map

    # map_stop = stop_weights.reshape((-1, 1))
    map_start = map_start.tocsc()
    map_stop = map_stop.tocsc()

    flattened_start = start.reshape((num_per_step, -1))
    flattened_stop = stop.reshape((num_per_step, -1))
    mapped_start_array = dot(map_start, flattened_start)
    mapped_stop_array = dot(map_stop, flattened_stop)
    
    if combine_input is None:
        combine_input = _check_whether_to_combine_inputs(start, stop)
    
    offset_map = None
    if combine_input:
        map = mapped_start_array.linear_map + mapped_stop_array.linear_map
        if offset is not None:
            offset_map = mapped_start_array.offset_map + mapped_stop_array.offset_map
            # Also TODO: consider storing a sparse matrix of zeros for offsets instead of none to make logic gates easier to work around.
        input = start.input
    else:
        map = sps.hstack((mapped_start_array.linear_map, mapped_stop_array.linear_map))
        if offset is not None:
            offset_map = np.vstack((mapped_start_array.offset_map, mapped_stop_array.offset_map)) + offset # TODO probably need to type-check between numpy and scipy
            # Also TODO: consider storing a sparse matrix of zeros for offsets instead of none to make logic gates easier to work around.
        input = np.vstack((start.input, stop.input))

    new_shape = (num_steps,) + tuple(start.shape)

    new_array = array(input=input, linear_map=map, offset=offset_map, shape=new_shape)

    return new_array


def linspace(start, stop, num_steps:int=50, combine_input:bool=None, offset=None):
    '''
    Perform a linear combintation between two arrays.
    The input is the input of the start array vertically stacked with the input
    of the stop array unless if the inputs are combined. The inputs are
    combined if combine_input=True or if combine_input=None and the start and
    stop have the same input.

    Parameters
    ----------
    start: array_like
        The first array that will be used in the combination.

    stop: array_like
        The second array.

    num_steps: int
        The number of steps in the combination.

    combine_input : boolean
        A boolean on whether the two inputs share their inputs. If so, the inputs are merged for this array.

    offset: np.ndarray
        A constant offset applied after the linear evaluation.
    '''

    #dims_dont_match = pointset_start.shape != pointset_end.shape
    # if any(dims_dont_match): #pset1 and pset2 must have same number of points
    #     print('The sets you are trying to interpolate do not have the same dimensions.\n')
    #     return

    # TODO Add checking to make sure function inputs are correct sizes.

    if num_steps == 1:
        stop_weights = np.array([0.5])
    else:
        stop_weights = np.arange(num_steps)/(num_steps-1)
    start_weights = 1 - stop_weights

    return linear_combination(start, stop, num_steps=num_steps,
            start_weights=start_weights, stop_weights=stop_weights, combine_input=combine_input, 
            offset=offset)



def _num_elements(x):
    if len(x.shape) == 1 :
        return 1
    else:
        return np.cumprod(x.shape[:-1])[-1]

def _arrays_are_equal(a:np.ndarray, b:np.ndarray):
    difference = a - b
    if np.any(difference):
        return False
    else:
        return True

def _check_whether_to_combine_inputs(input1:MappedArray, input2:MappedArray):
    if _arrays_are_equal(input1.input, input2.input):
        return True
    else:
        return False 


if __name__ == "__main__":
    input = np.array([1, 2, 3])
    a = MappedArray([1, 2, 3])  # this would be am.MappedArray(...) or am.array(...)

    print('a', a)

    map = np.arange(9).reshape((3,3))

    b = dot(map, a)     # the dot would be am.dot
    b_copy = dot(map, a)
    print('b', b)

    print('testing subtraction with MappedArray', b - b_copy)
    print('testing subtraction with numpy', b - np.dot(map, input))
    print('numpy_check (should be 0)', np.linalg.norm(b.value - np.dot(map, input)))
    print('dotting a numpy array check', np.linalg.norm(dot(map, input).value - np.dot(map, input)))

    map2 = np.arange(12).reshape((4,3))
    c = dot(map2, a)
    print('c', c)
    print('c.shape', c.shape)
    print('c_numpy_check', np.linalg.norm(c.value - np.dot(map2, input)))

    d = dot(map2, b)
    print('d', d)
    print('d_numpy_check', np.linalg.norm(d.value - map2.dot(map).dot(input)))

    input2 = np.array([2, 4, 6])
    print('d2', d.evaluate(input2))
    print('d2_numpy_check', np.linalg.norm(d.evaluate(input2) - map2.dot(map).dot(input2)))


    f = linspace(a, b, num_steps=8, combine_input=None)
    print('array_mapper linspace: ', f)
    print('linspace_numpy_check', np.linalg.norm(f.value - np.linspace(input, np.dot(map, input), num=8)))
