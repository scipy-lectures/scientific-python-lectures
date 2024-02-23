"""
Analysis of Iris petal and sepal sizes
=======================================

Illustrate an analysis on a real dataset:

- Visualizing the data to formulate intuitions
- Fitting of a linear model
- Hypothesis test of the effect of a categorical variable in the presence
  of a continuous confound

"""

import matplotlib.pyplot as plt

import pandas
from pandas import plotting

from statsmodels.formula.api import ols

# Load the data
data = pandas.read_csv("iris.csv")

##############################################################################
# Plot a scatter matrix

# Express the names as categories
categories = pandas.Categorical(data["name"])

# The parameter 'c' is passed to plt.scatter and will control the color
plotting.scatter_matrix(data, c=categories.codes, marker="o")

fig = plt.gcf()
fig.suptitle("blue: setosa, green: versicolor, red: virginica", size=13)

##############################################################################
# Statistical analysis

# Let us try to explain the sepal length as a function of the petal
# width and the category of iris

model = ols("sepal_width ~ name + petal_length", data).fit()
print(model.summary())

# Now formulate a "contrast", to test if the offset for versicolor and
# virginica are identical

print("Testing the difference between effect of versicolor and virginica")
print(model.f_test([0, 1, -1, 0]))
plt.show()
