"""
Measuring Decision Tree performance
====================================

Demonstrates overfit when testing on train set.
"""

############################################################
# Get the data

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)

############################################################
# Train and test a model
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor().fit(data.data, data.target)

predicted = clf.predict(data.data)
expected = data.target

############################################################
# Plot predicted as a function of expected

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 5], [0, 5], "--k")
plt.axis("tight")
plt.xlabel("True price ($100k)")
plt.ylabel("Predicted price ($100k)")
plt.tight_layout()

############################################################
# Pretty much no errors!
#
# This is too good to be true: we are testing the model on the train
# data, which is not a measure of generalization.
#
# **The results are not valid**
