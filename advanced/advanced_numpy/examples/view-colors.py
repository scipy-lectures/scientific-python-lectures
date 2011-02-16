x = np.zeros((10, 10, 4), dtype=np.int8)
x[:,:,0] = 1
x[:,:,1] = 2
x[:,:,2] = 3
x[:,:,3] = 4

# How to make a (10, 10) structured array with fields 'r', 'g', 'b', 'a',
# without copying?

y = ...

assert (y['r'] == 1).all()
assert (y['g'] == 2).all()
assert (y['b'] == 3).all()
assert (y['a'] == 4).all()
