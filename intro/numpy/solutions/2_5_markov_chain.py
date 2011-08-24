import numpy as np

np.random.seed(1234)

n_states = 5
n_steps = 50
tolerance = 1e-5

# Random transition matrix and state vector
P = np.random.rand(n_states, n_states)
p = np.random.rand(n_states)

# Normalize rows in P
P /= P.sum(axis=1)[:,np.newaxis]

# Normalize p
p /= p.sum()

# Take steps
for k in xrange(n_steps):
    p = P.T.dot(p)

p_50 = p
print p_50

# Compute stationary state
w, v = np.linalg.eig(P.T)

j_stationary = np.argmin(abs(w - 1.0))
p_stationary = v[:,j_stationary].real
p_stationary /= p_stationary.sum()
print p_stationary

# Compare
if all(abs(p_50 - p_stationary) < tolerance):
    print "Tolerance satisfied in infty-norm"

if np.linalg.norm(p_50 - p_stationary) < tolerance:
    print "Tolerance satisfied in 2-norm"
