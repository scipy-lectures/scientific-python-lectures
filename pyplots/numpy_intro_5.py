import numpy as np
import matplotlib.pyplot as plt

n_stories = 1000 # number of walkers
t_max = 200      # time during which we follow the walker

# We randomly choose all the steps 1 or -1 of the walk

t = np.arange(t_max)
steps = 2 * np.random.random_integers(0, 1, (n_stories, t_max)) - 1
print np.unique(steps) # Verification: all steps are 1 or -1
# [-1,  1]

# We build the walks by summing steps along the time

positions = np.cumsum(steps, axis=1) # axis = 1: dimension of time
sq_distance = positions**2

# We get the mean in the axis of the stories

mean_sq_distance = np.mean(sq_distance, axis=0)

# Plot the results:

plt.figure(figsize=(4, 3))
plt.plot(t, np.sqrt(mean_sq_distance), 'g.', t, np.sqrt(t), 'y-')
plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")
plt.show()
