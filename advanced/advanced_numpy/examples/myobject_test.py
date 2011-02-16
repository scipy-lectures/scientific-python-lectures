#
# Compile myobject.c first with
#
#    python3 setup_myobject.py build_ext -i
#
# If you are interested, play a bit with changing things in ``myobject.c``
#

import myobject

obj = myobject.MyObject()
view = memoryview(obj)

print("shape", view.shape)
print("strides", view.strides)
print("format", view.format)


#
# If you also have Numpy for Python 3 ...
#

import numpy as np

x = np.asarray(obj)
print(x)

# this prints
#
# [[1 2]
#  [3 4]]
