"""
Test for an education/gender interaction in wages
==================================================

Wages depend mostly on education. Here we investigate how this dependence
is related to gender: not only does gender create an offset in wages, it
also seems that wages increase more with education for males than
females.

Does our data support this last hypothesis? We will test this using
statsmodels' formulas
(http://statsmodels.sourceforge.net/stable/example_formulas.html).

"""

##############################################################################
# Load and massage the data
import pandas

import urllib
import os

if not os.path.exists('wages.txt'):
    # Download the file if it is not present
    urllib.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                       'wages.txt')

# EDUCATION: Number of years of education
# SEX: 1=Female, 0=Male
# WAGE: Wage (dollars per hour)
data = pandas.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None,
                       header=None, names=['education', 'gender', 'wage'],
                       usecols=[0, 2, 5],
                       )

# Convert genders to strings (this is particulary useful so that the
# statsmodels formulas detects that gender is a categorical variable)
import numpy as np
data['gender'] = np.choose(data.gender, ['male', 'female'])

# Log-transform the wages, because they typically are increased with
# multiplicative factors
data['wage'] = np.log10(data['wage'])


##############################################################################
# simple plotting
import seaborn

# Plot 2 linear fits for male and female.
seaborn.lmplot(y='wage', x='education', hue='gender', data=data)


##############################################################################
# statistical analysis
import statsmodels.formula.api as sm

# Note that this model is not the plot displayed above: it is one
# joined model for male and female, not separate models for male and
# female. The reason is that a single model enables statistical testing
result = sm.ols(formula='wage ~ education + gender', data=data).fit()
print(result.summary())


##############################################################################
# The plots above highlight that there is not only a different offset in
# wage but also a different slope
#
# We need to model this using an interaction
result = sm.ols(formula='wage ~ education + gender + education * gender',
                data=data).fit()
print(result.summary())


##############################################################################
# Looking at the p-value of the interaction of gender and education, the
# data does not support the hypothesis that education benefits males
# more than female (p-value > 0.05).


import matplotlib.pyplot as plt
plt.show()

