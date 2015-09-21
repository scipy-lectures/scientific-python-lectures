"""
Air fares before and after 9/11
=====================================

This is a business-intelligence (BI) like application.

What is interesting here is that we may want to study fares as a function
of the year, paired accordingly to the trips, or forgetting the year,
only as a function of the trip endpoints.

Using statsmodels' linear models, we find that both with an OLS (ordinary
least square) and a robust fit, the intercept and the slope are
significantly non-zero: the air fares have decreased between 2000 and
2001, and their dependence on distance travelled has also decreased

"""

# Standard library imports
import urllib
import os

##############################################################################
# Load the data
import pandas

if not os.path.exists('airfares.txt'):
    # Download the file if it is not present
    urllib.urlretrieve(
        'http://www.stat.ufl.edu/~winner/data/airq4.dat',
                       'airfares.txt')

# As a seperator, ' +' is a regular expression that means 'one of more
# space'
data = pandas.read_csv('airfares.txt', sep=' +', header=0,
                       names=['city1', 'city2', 'pop1', 'pop2',
                              'dist', 'fare_2000', 'nb_passengers_2000',
                              'fare_2001', 'nb_passengers_2001'])

# we log-transform the number of passengers
import numpy as np
data['nb_passengers_2000'] = np.log10(data['nb_passengers_2000'])
data['nb_passengers_2001'] = np.log10(data['nb_passengers_2001'])

##############################################################################
# Make a dataframe whith the year as an attribute, instead of separate columns

# This involves a small danse in which we separate the dataframes in 2,
# one for year 2000, and one for 2001, before concatenating again.

# Make an index of each flight
data_flat = data.reset_index()

data_2000 = data_flat[['city1', 'city2', 'pop1', 'pop2',
                       'dist', 'fare_2000', 'nb_passengers_2000']]
# Rename the columns
data_2000.columns = ['city1', 'city2', 'pop1', 'pop2', 'dist', 'fare',
                     'nb_passengers']
# Add a column with the year
data_2000['year'] = 2000

data_2001 = data_flat[['city1', 'city2', 'pop1', 'pop2',
                       'dist', 'fare_2001', 'nb_passengers_2001']]
# Rename the columns
data_2001.columns = ['city1', 'city2', 'pop1', 'pop2', 'dist', 'fare',
                     'nb_passengers']
# Add a column with the year
data_2001['year'] = 2001

data_flat = pandas.concat([data_2000, data_2001])


##############################################################################
# Plot scatter matrices highlighting different aspects

import seaborn
seaborn.pairplot(data_flat, vars=['fare', 'dist', 'nb_passengers'],
                 kind='reg', markers='.')

# A second plot, to show the effect of the year (ie the 9/11 effect)
seaborn.pairplot(data_flat, vars=['fare', 'dist', 'nb_passengers'],
                 kind='reg', hue='year', markers='.')


##############################################################################
# Plot the difference in fare

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 2))
seaborn.boxplot(data.fare_2001 - data.fare_2000)
plt.title('Fare: 2001 - 2000')
plt.subplots_adjust()

plt.figure(figsize=(5, 2))
seaborn.boxplot(data.nb_passengers_2001 - data.nb_passengers_2000)
plt.title('NB passengers: 2001 - 2000')
plt.subplots_adjust()


##############################################################################
# Statistical testing: dependence of fare on distance and number of
# passengers
import statsmodels.formula.api as sm

result = sm.ols(formula='fare ~ 1 + dist + nb_passengers', data=data_flat).fit()
print(result.summary())

# Using a robust fit
result = sm.rlm(formula='fare ~ 1 + dist + nb_passengers', data=data_flat).fit()
print(result.summary())


##############################################################################
# Statistical testing: regression of fare on distance: 2001/2000 difference

result = sm.ols(formula='fare_2001 - fare_2000 ~ 1 + dist', data=data).fit()
print(result.summary())

# Plot the corresponding regression
data['fare_difference'] = data['fare_2001'] - data['fare_2000']
seaborn.lmplot(x='dist', y='fare_difference', data=data)

plt.show()

