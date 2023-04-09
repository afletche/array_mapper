import numpy as np
import scipy.sparse as sps
import scipy
import csdl
from python_csdl_backend import Simulator

def array(input=None, linear_map=None, offset=None, shape:tuple=None):
    '''
    Creates a MappedArray.

    Parameters
    ----------
    input: array_like
        The input into the map to calculate the array

    linear_map: numpy.ndarray or scipy.sparse matrix
        Linear map for evaluating the array
    
    offset: numpy.ndarray
        Offset for evaluating the array

    shape: tuple
        The shape of the MappedArray
    '''
    return MappedArray(input=input, linear_map=linear_map, offset=offset, shape=shape)


class MappedArray:
    '''
    An instance of a MappedArray object. This is an array that stores an affine mapping to calculate itself

    Parameters
    ----------
    input: array_like
        The input into the map to calculate the array

    linear_map: numpy.ndarray or scipy.sparse matrix
        Linear map for evaluating the array
    
    offset: numpy.ndarray
        Offset for evaluating the array

    shape: tuple
        The shape of the MappedArray
    '''
    
    def __init__(self, input=None, linear_map=None, offset=None, shape:tuple=None) -> None:
        '''
        Creates an instance of a MappedArray object.

        Parameters
        ----------
        input: array_like
            The input into the map to calculate the array

        linear_map: numpy.ndarray or scipy.sparse matrix
            Linear map for evaluating the array
        
        offset: numpy.ndarray
            Offset for evaluating the array

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
            # elif type(input) is CSDLVariable:
            #     return csdl_variable_with_csdl_operation_done
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
    


class CSDLEvaluationModel(csdl.Model):
    '''
    Creates a CSDL model that can take in a hanging input.
    '''

    def initialize(self):
        self.parameters.declare('csdl_model')
        self.parameters.declare('input')

    def define(self):
        csdl_model = self.parameters['csdl_model']
        input = self.parameters['input']

        if input is not None:
            self.create_input('input', val=input)
        self.add(submodel=csdl_model, name='csdl_model')
    

class NonlinearMappedArray:
    '''
    A NonlinearMappedArray object. This map can be nonlinear.

    Parameters
    ----------
    input: {np.ndarray, MappedArray}
        The input to the nonlinear map to calculate this array

    csdl_model: csdl.Model
        A CSDL model that contains the nonlinear mapping. The CSDL model must have a declare_variable("input",...) and outputs "output"
    '''
    
    def __init__(self, input=None, csdl_model:csdl.Model=None) -> None:
        '''
        Creates an instance of a NonlinearMappedArray object.

        Parameters
        ----------
        input: {np.ndarray, MappedArray}
            The input to the nonlinear map to calculate this array

        csdl_model: csdl.Model
            A CSDL model that contains the nonlinear mapping. The CSDL model must have a declare_variable("input",...) and outputs "output"
        '''
        
        # Listing list of attributes
        self.input = input
        self.csdl_model = csdl_model
        self.value = None

        self.evaluate()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'array_mapper.NonlinearMappedArray(input={self.input}, csdl_model={self.csdl_model})'

    def __pos__(self):
        return self

    # TODO: It seems like it should add a *-1 operation to end of CSDL model
    # def __neg__(self):
    #     map = None
    #     offset = None
    #     input = self.input
    #     if self.linear_map is not None:
    #         map = -self.linear_map
    #     else:
    #         input = -self.input

    #     if self.offset_map is not None:
    #         offset = -self.offset_map

    #     return array(input=input, linear_map=map, offset=offset, shape=self.shape)

    # TODO add addition operation to CSDL model
    # def __add__(self, x2):
    #     return add(self, x2)

    # def __radd__(self, x1):
    #     return add(self, x1)

    # def __sub__(self, x2):
    #     return add(self, -x2)

    # def __rsub__(self, x1):
    #     return add(-self, x1)

    # def __mul__(self, alpha):
    #     map = None
    #     offset = None
    #     if self.linear_map is not None:
    #         map = alpha*self.linear_map
    #     if self.offset_map is not None:
    #         offset = alpha*self.offset_map

    #     return array(input=self.input, linear_map=map, offset=offset, shape=self.shape)

    # def __rmul__(self, alpha):
    #     return self.__mul__(alpha)

    # def __truediv__(self, alpha):
    #     return self.__mul__(1/alpha)

    # TODO this adds a reshape operation to the CSDL model
    # def reshape(self, newshape):
    #     new_array = MappedArray(input=self.input, linear_map=self.linear_map, offset=self.offset_map, shape=newshape)
    #     return new_array

    def evaluate(self, input=None):
        '''
        Evaluate the value given an input. If the stored input is a MappedArray, the MappedArray is evaluated using the supplied input
        '''
        if type(self.input) is MappedArray:
            self.input.evaluate(input)
            input = self.input.value
        elif input is None and self.input is not None:
            input = self.input
        elif input is None and self.input is None:
            return
        elif type(input) is MappedArray:
            input.evaluate()
            input = input.value
            self.input = input

        evaluation_model = CSDLEvaluationModel(csdl_model=self.csdl_model, input=input)

        sim = Simulator(evaluation_model)
        sim.run()

        self.value = sim['output']
        self.shape = self.value.shape

        return self.value
    
    # TODO: Implement these so they can be used for design geometry optimization
    # def evaluate_first_derivative(self,input):
    #     pass

    # def evaluate_second_derivative(self,input):
    #     pass


def vstack(tup:tuple, combinte_inputs:bool=None):
    '''
    Stacks a tuple of arrays.

    Note: This function is currently not smart enough to combine inputs for non-adjacent elements in the stack.

    Parameters
    -----------
    tup: tuple
        The tuple of arrays to be stacked
    combine_inputs: bool
        If both arguments are MappedArrays, this boolean determines whether
        the output MappedArray should stack the inputs of x1 and x2 as its
        input (False), or whether the output MappedArray should use the same
        input (True). None option automatically detects if arguments share
        same input. If so, inputs are combined.
    '''

    num_elements = len(tup)
    elements = list(tup)
    combine_inputs_list = []
    # if any numpy arrays are being stacked, convert them to MappedArrays with Identity map
    for i in range(num_elements):
        if type(elements[i]) is np.ndarray:
            elements[i] = MappedArray(elements[i], shape=elements[i].shape)

    # check whether inputs should be combined for each combination
    if combinte_inputs is None:
        for i in range(num_elements-1):
            combine_inputs_list.append(_check_whether_to_combine_inputs(tup[i], tup[i+1]))
    
    for i in range(num_elements-1):
        element1 = elements[i]
        element2 = elements[i+1]
        axis_0_length = element1.shape[0] + element2.shape[0]
        new_shape = (axis_0_length,) + element1.shape[1:]

        element1_offset_map = element1.offset_map
        element2_offset_map = element2.offset_map
        if element1.offset_map is None and element2.offset_map is not None:
            element1_offset_map = sps.csc_matrix(element1.shape)
        elif element1.offset_map is not None and element2.offset_map is None:
            element2_offset_map = sps.csc_matrix(element2.shape)
        offset_map = _vstack_maps(element1_offset_map, element2_offset_map)

        if combine_inputs_list[i]:
            linear_map = _vstack_maps(element1.linear_map, element2.linear_map)
            input = element1.input
        else:
            linear_map = _diag_stack_maps(element1.linear_map, element2.linear_map)
            input = np.append(element1.input, element2.input)

        elements[i+1] = MappedArray(input=input, linear_map=linear_map, offset=offset_map, shape=new_shape)

    return elements[-1]


def add(x1, x2, combine_inputs:bool=None):
    '''
    Adds the two arguments.

    Parameters
    -----------
    x1: array_like
        The first argument being added
    x2: array_like
        The second argument being added
    combine_inputs: bool
        If both arguments are MappedArrays, this boolean determines whether
        the output MappedArray should stack the inputs of x1 and x2 as its
        input (False), or whether the output MappedArray should use the same
        input (True). None option automatically detects if arguments share
        same input. If so, inputs are combined.
    '''
    if type(x1) is not MappedArray and type(x2) is not MappedArray:
        return x1 + x2
    elif type(x1) is not MappedArray:
        return add(x2=x2, x1=x1, combine_inputs=combine_inputs)

    map = None
    offset_map = None
    input = None

    if type(x2) is MappedArray:
        combine_inputs = _check_whether_to_combine_inputs(x1, x2)

        # if x1.linear_map is not None and x2.linear_map is not None:
        if combine_inputs:
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

        if combine_inputs:
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
    combine_inputs: bool
        If both arguments are MappedArrays, this boolean determines whether
        the output MappedArray should stack the inputs of x1 and x2 as its
        input (False), or whether the output MappedArray should use the same
        input (True). None option automatically detects if arguments share
        same input. If so, inputs are combined.
    '''
    return add(x1, -x2, combine_inputs=combine_inputs)


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
    if type(map) is MappedArray:
        map = map.value    # NOTE: This is throwing away previous computations becase that would be nonlinear.

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



def linear_combination(start, stop, num_steps=50, start_weights=None, stop_weights=None, combine_inputs=None, offset=None):
    '''
    Perform a linear combintation between two arrays.
    The input is the input of the start array vertically stacked with the input
    of the stop array unless if the inputs are combined. The inputs are
    combined if combine_inputs=True or if combine_inputs=None and the start and
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

    combine_inputs : boolean
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
        linspace(start, stop, num_steps, combine_inputs, offset)

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
    
    if combine_inputs is None:
        combine_inputs = _check_whether_to_combine_inputs(start, stop)
    
    offset_map = None
    if combine_inputs:
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


def linspace(start, stop, num_steps:int=50, combine_inputs:bool=None, offset=None):
    '''
    Perform a linear combintation between two arrays.
    The input is the input of the start array vertically stacked with the input
    of the stop array unless if the inputs are combined. The inputs are
    combined if combine_inputs=True or if combine_inputs=None and the start and
    stop have the same input.

    Parameters
    ----------
    start: array_like
        The first array that will be used in the combination.

    stop: array_like
        The second array.

    num_steps: int
        The number of steps in the combination.

    combine_inputs : boolean
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
            start_weights=start_weights, stop_weights=stop_weights, combine_inputs=combine_inputs, 
            offset=offset)



class NormModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('x')
        self.parameters.declare('ord')
        self.parameters.declare('axes')

    def define(self):
        x = self.parameters['x']
        ord = self.parameters['ord']
        axes = self.parameters['axes']
        input_csdl = self.declare_variable('input', shape=x.shape)
        if len(x.shape) == len(axes):
            self.register_output('output', csdl.pnorm(input_csdl, pnorm_type=ord))
        else:
            self.register_output('output', csdl.pnorm(input_csdl, axis=axes, pnorm_type=ord))

def norm(x, ord=2, axes:tuple=(-1,)):
    '''
    x: {MappedArray, NonlinearMappedArray}
        The array that is the input to the norm

    ord: {non-zero int, inf, -inf, "fro", "nuc"}, optional
        Order of the norm (in a p-norm sense)

    axis: int
        axis along which the norm will be taken
    '''
    # norm_model = csdl.Model()
    # input_csdl = norm_model.declare_variable('input', shape=x.shape)
    # norm_model.register_output('output', csdl.pnorm(input_csdl, pnorm_type=ord))

    # NOTE: need to account for operations that happen in input
    # nonlinear_mapped_array = NonlinearMappedArray(input=x, csdl_model=norm_model)
    nonlinear_mapped_array = NonlinearMappedArray(input=x, csdl_model=NormModel(x=x, ord=ord, axes=axes))
    return nonlinear_mapped_array


def _num_elements(x):
    if len(x.shape) == 1 :
        return x.shape[0]
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
    
def _vstack_maps(map1, map2):
    if map1 is None and map2 is None:
        return None

    if type(map1) is np.ndarray and type(map2) is np.ndarray:
        return np.vstack((map1,map2))
    elif sps.isspmatrix(map1) and sps.isspmatrix(map2):
        return sps.vstack((map1,map2))
    elif type(map1) is np.ndarray and sps.isspmatrix(map2):
        return np.vstack((map1, map2.toarray()))
    elif sps.isspmatrix(map1) and type(map2) is np.ndarray:
        return np.vstack((map1.toarray(), map2))
    else:
        return Exception("Array Mapper is trying to vertically stack maps that aren't numpy arrays or scipy sparse matrices")
    
def _diag_stack_maps(map1, map2):
    if type(map1) is np.ndarray and type(map2) is np.ndarray:
        return scipy.linalg.block_diag(map1,map2)
    elif sps.isspmatrix(map1) and sps.isspmatrix(map2):
        return sps.block_diag((map1,map2))
    elif type(map1) is np.ndarray and sps.isspmatrix(map2):
        return scipy.linalg.block_diag(map1, map2.toarray())
    elif sps.isspmatrix(map1) and type(map2) is np.ndarray:
        return scipy.linalg.block_diag(map1.toarray(), map2)
    else:
        return Exception("Array Mapper is trying to diagonally stack maps that aren't numpy arrays or scipy sparse matrices")


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


    f = linspace(a, b, num_steps=8, combine_inputs=None)
    print('array_mapper linspace: ', f)
    print('linspace_numpy_check', np.linalg.norm(f.value - np.linspace(input, np.dot(map, input), num=8)))

    g = vstack((b,c))
    print('array_mapper vstack', g)

    h0 = c-d/10
    print('c-d/10', h0)
    h = norm(h0)
    print('norm(c-d/10)', h)
    print('norm numpy check 1', np.linalg.norm(h0.value) - h.value)
    new_input_c = b.value/8
    new_input_d = d.input/2
    new_input = np.append(new_input_c, new_input_d)
    h.evaluate(new_input)
    print('new c-d/10', h0.evaluate(new_input))
    print('new norm(c-d/10)', h)
    print('norm numpy check after new evaluation', np.linalg.norm(h0.value) - h.value)
    i = np.arange(6).reshape((2,3))
    print('i', i)
    print('norm(i)', norm(i, ord=2, axes=(0,1)))

