import numpy as np

class MappedArray:
    ''' Sketch of MappedArray class for purpose of developing sample run script '''
    
    def __init__(self, object=None, shape=None) -> None:
        '''
        Creates an instance of a MappedArray object.

        Parameters
        ----------
        object: array_like
            The value of the MappedArray

        shape: tuple
            The shape of the MappedArray
        '''
        
        # Listing list of attributes
        self.linear_map = None
        self.offset_map = None
        self.shape = None
        self.input = None
        self.value = None

        if type(object) is np.ndarray:
            self.input = object
            self.value = object
            self.shape = object.shape
        elif type(object) is list or type(object) is tuple:
            self.input = np.array(object)
            self.value = np.array(object)
            self.shape = np.array(object).shape
        elif type(object) is MappedArray:   # Creates a copy of the input MappedArray
            self.linear_map = object.linear_map
            self.offset_map = object.offset_map
            self.shape = object.shape
            self.input = object.input
            self.value = object.value

        if type(shape) is list or type(shape) is tuple:
            self.shape = shape

    def __str__(self):
        return str(self.value)

    
    def __add__(self, x2):
        new_array = MappedArray(object=self.input)
        if type(x2) is MappedArray:
            new_array.linear_map = self.linear_map + x2.linear_map
            new_array.offset_map = self.offset_map + x2.offset_map
        elif type(x2) is np.ndarray:
            new_array.offset_map = self.offset_map + x2

        return new_array

    def evaluate(self, input=None):
        if input is not None:
            self.input = input
        if self.linear_map is not None and self.offset_map is not None:
            self.value = self.linear_map.dot(self.input) + self.offset_map
        elif self.linear_map is not None:
            self.value = self.linear_map.dot(self.input)
        elif self.offset_map is not None:
            self.value = self.input + self.offset_map
        else:
            self.value = self.input

        return self.value