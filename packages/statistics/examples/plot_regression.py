"""
Simple Regression
====================

Fit a simple linear regression using 'statsmodels', compute corresponding
p-values.
"""

# Original author: Thomas Haslwanter

import numpy as np
import matplotlib.pyplot as plt
import pandas

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols

# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm

##############################################################################
# Generate and show the data
x = np.linspace(-5, 5, 20)

# To get reproducible values, provide a seed value
rng = np.random.default_rng(27446968)

y = -5 + 3 * x + 4 * np.random.normal(size=x.shape)

# Plot the data
plt.figure(figsize=(5, 4))
plt.plot(x, y, "o")

##############################################################################
# Multilinear regression model, calculating fit, P-values, confidence
# intervals etc.

# Convert the data into a Pandas DataFrame to use the formulas framework
# in statsmodels
data = pandas.DataFrame({"x": x, "y": y})

# Fit the model
model = ols("y ~ x", data).fit()

# Print the summary
print(model.summary())

# Perform analysis of variance on fitted linear model
anova_results = anova_lm(model)

print("\nANOVA results")
print(anova_results)

##############################################################################
# Plot the fitted model

# Retrieve the parameter estimates
offset, coef = model._results.params
plt.plot(x, x * coef + offset)
plt.xlabel("x")
plt.ylabel("y")

plt.show()
