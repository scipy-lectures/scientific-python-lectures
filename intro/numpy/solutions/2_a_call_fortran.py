import numpy as np
import fortran_module

def some_function(input):
    """
    Call a Fortran routine, and preserve input shape
    """
    input = np.asarray(input)
    # fortran_module.some_function() only accepts 1-D arrays!
    output = fortran_module.some_function(input.ravel())
    return output.reshape(input.shape)

print some_function(np.array([1, 2, 3]))
print some_function(np.array([[1, 2], [3, 4]]))
