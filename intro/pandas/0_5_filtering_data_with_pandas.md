---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Filtering Data Frames with Boolean Series

<!---
📝 NOTE: from meeting with Matthew: Boolean series (and matching of row labels there) e.g. Boolean array will operate like iloc, Boolean Series will match row labels.

Add a separate page for Boolean indexing in a sensible place (build with numpy first first then show with Series) - show that order doesn't matter e.g. Booleans are matched by label not by position. Show this error:

```
import pandas as pd
df = pd.read_csv('birth_weights.csv')
df.head()
female_mf = df['sex'] == 2
female_mf.head()
df2 = df.sort_values('sex')
df2
df2[female_mf].describe()
df[female_mf].describe()
df.loc[female_mf]
df.iloc[female_mf]
```

Covered on this page:

* filtering (ex: df[df['age'] > 30])
* Boolean indexing, with Boolean series (and matching of row labels there); range indexing on labels (inclusive of last element)

Structure of this page:

* show Boolean arrays and boolean filtering of arrays

* build a Series, show Boolean series and filtering Series with Boolean arrays/series

* show Boolean filtering with DataFrames
-->

Once we have got our data into a Pandas Data Frame, no doubt we will want to filter and select specific portions of it for visualisation and analysis. 

On this page we will look at different methods of filtering. As on previous pages, we will focus on how Pandas is built from Numpy, and assess the similarities and differences in how filtering works on objects from both libraries.

+++

## Pandas from Numpy, once more

To explore filtering, let's create some Numpy arrays, containing the [HDI data](https://ourworldindata.org/grapher/children-per-woman-vs-human-development-index):

```{code-cell} ipython3
# Import libraries for this page
import numpy as np
import pandas as pd
```

```{code-cell} ipython3
# Set an option to match `.replace` behavior to upcoming Pandas 3.
# See <https://github.com/pandas-dev/pandas/issues/57734> for a discussion.
# We do this to avoid confusing warnings for one solution to the exercise below.
pd.set_option('future.no_silent_downcasting', True)
```

We build the constituent arrays:

```{code-cell} ipython3
# Standard three-letter code for each country
country_codes_array = np.array(['AUS', 'BRA', 'CAN',
                                'CHN', 'DEU', 'ESP',
                                'FRA', 'GBR', 'IND',
                                'ITA', 'JPN', 'KOR',
                                'MEX', 'RUS', 'USA'])
country_codes_array
```

```{code-cell} ipython3
# Human Development Index Scores for each country
hdis_array = np.array([0.896, 0.668, 0.89,
                       0.586, 0.89,  0.828,
                       0.844, 0.863, 0.49,
                       0.842, 0.883, 0.824,
                       0.709, 0.733, 0.894])
hdis_array
```

Let's say we want to view HDI values that are bigger than the median HDI
value.

First, we can calculate the median, using the `np.median()` function:

```{code-cell} ipython3
# Get the median HDI
np.median(hdis_array)
```

Next, we can create a *Boolean array* by using a comparison operator.

In this case `>` is the operator we want, with the `hdis_array` on the left hand side, and the median value of the array on the right hand side:

```{code-cell} ipython3
# Create a Boolean array
hdis_array_grtr_median_bool = hdis_array > np.median(hdis_array)
hdis_array_grtr_median_bool
```

The array in the output of the cell above is the same length as the
`hdis_array` and it contains `True` value where the element in `hdis_array` is
larger than the median HDI score (and `False` values otherwise).

+++

Our Boolean array can now be used to retrieve elements of the `hdis_array` that are greater than the median HDI value. We just place our filter in between square brackets used to index the array. The `True` values act as "on switches" (keeping that element in the output array) and the `False` values act as "off switches" (removing that element from the output array):

```{code-cell} ipython3
# Filter the `hdis_array`
# Notice that this is direct indexing.
hdis_array[hdis_array_grtr_median_bool]
```

If we're particularly neurotic, we can use the `np.all` function that this operation has, in fact, only returned elements greater than the median:

```{code-cell} ipython3
# Check that all remaining values are in fact above the median.
np.all(hdis_array[hdis_array_grtr_median_bool] > np.median(hdis_array))
```

Because Python and Numpy treat `True` values as being equal to 1 and `False` values as being equal to 0, taking the sum of a Boolean array calculates the number of True values in the array:

```{code-cell} ipython3
n_trues = np.sum(hdis_array_grtr_median_bool)
n_trues
```

If we divide by the number of elements in the Boolean array, we have proportion of elements that are True:

```{code-cell} ipython3
n_trues / len(hdis_array_grtr_median_bool)
```

We do the same calculation with `np.mean()`.  `np.mean` on a Boolean array is a good way to count the proportion of `True` values. In our case it is fairly close to 0.5, as we might expect, because we are compared to the median

```{code-cell} ipython3
# Count the proportion of `True` values
np.mean(hdis_array_grtr_median_bool)
```

This can be very useful when counting the proportion of participants or observations with a particular categorical feature (proportion of males/females, proportion of democratic countries etc).

We can also, if we wish, turn our Boolean array into a Boolean *Series* using the `pd.Series()` constructor.

The Series can benefit from an `.index`. In the present context, when we put the three-letter country codes as the `index`, we can keep track of the country that each filtered value corresponds to.

Let's create a Boolean Series:

```{code-cell} ipython3
# Make a Boolean Series from the Boolean array.
hdis_array_bool_series = pd.Series(hdis_array_grtr_median_bool, 
                                   index=country_codes_array)
hdis_array_bool_series
```

Even better, let's manually specify the `.name` attribute - to make it clearer
down the line what the `True` and `False` values mean — i.e. whether each
country has a higher-than-median HDI score:

```{code-cell} ipython3
# Set the `.name` attribute.
hdis_array_bool_series.name = 'hdi_grtr_thn_median'
hdis_array_bool_series
```

This has advantages over the Numpy Boolean array in terms of interpretability: it is less easy to forget the meaning of what is in the `values` component of the Series!

We also have access to all of the Pandas Series methods:

```{code-cell} ipython3
# Max (True counts as 1).
hdis_array_bool_series.max()
```

```{code-cell} ipython3
# Min (False counts as 1).
hdis_array_bool_series.min()
```

```{code-cell} ipython3
# Number of True values.
hdis_array_bool_series.sum()
```

```{code-cell} ipython3
# Mean (returns the proportion of `True` values)
hdis_array_bool_series.mean()
```

Boolean Series are particularly useful for filtering Series and Data Frames.

+++

## Filtering in Pandas

You know by now that *Data Frames are a dictionary-like collection of Series*.

Because Series are built (in part) from Numpy arrays, filtering the Data Frame works very much like filtering with Boolean Numpy arrays and Pandas Series.

As with Series, the real advantage comes in having a shared index, as it keeps the filtering process highly interpretable and makes it less prone to error.

To explore further, let's import the HDI data:

```{code-cell} ipython3
# Import our dataset
df = pd.read_csv("data/year_2000_hdi_fert.csv")
# Set the index
df = df.set_index("Code")
df
```

Now, to view the median we can use the `.median()` Series method, rather than calling the `np.median()` function as we did earlier.

*NB*: we mentioned on a [previous page](0_2_pandas_dataframes_attributes_methods) that Pandas Series *methods* have parallel behavior to Numpy *functions*, but they generally give better efficiency and readability, as is the case here.  Remember too that they treat NaN values as *missing* rather than *numerically invalid*, so Pandas calculations, unlike Numpy's default calculations, typically drop NaN values.

```{code-cell} ipython3
# Get the median HDI
df['Human Development Index'].median()
```

Because Data Frames force all their constituent Series (columns) to share an `index`, creating a Boolean Series has the desirable characteristic of associating a row label with each Boolean.

For instance, let's use direct indexing with a column name (`df['Human Development Index']`) to create a Boolean Series which is `True` where the country has a HDI score above the median HDI score: 

```{code-cell} ipython3
# Create a Boolean Series
hdi_gt_median = df['Human Development Index'] > df['Human Development Index'].median()
hdi_gt_median
```

As mentioned above, this is more interpretable that a Numpy array just containing Booleans, with no `index`, because the `index` provides readable labels for the values.

Again, to increase interpretabiity, we can assign this Boolean Series a new `name` attribute:

```{code-cell} ipython3
# Create a Boolean Series by using a comparison operator on the DataFrame column
hdi_gt_median.name = 'HDI_grtr_than_median'
hdi_gt_median
```

We can now use our Boolean Series to filter the whole Data Frame. This let's use this filtering to select those rows corresponding to countries with greater than median HDI scores:

```{code-cell} ipython3
# Filter the DataFrame using the Boolean Series we got from using a comparison operator
# on the DataFrame column.
df[hdi_gt_median]
```

This filtering operation was done via a Boolean Series which was created using direct indexing on the Data Frame itself (`df['Human Development Index']`).

Earlier, we "hand built" a Boolean Series from a Numpy array, another array for the index, and a manually specified `name` attribute. Here is that Series:

```{code-cell} ipython3
# Our "handmade" Boolean Series.
hdis_array_bool_series
```

This hand-built Series returns the same rows of the Data Frame as Series we got by using only Data Frame columns:

```{code-cell} ipython3
# We can also filter with our "handmade" Series.
df[hdis_array_bool_series]
```

We can use the Boolean Series to create a new Data Frame. This lets us use Pandas' statistical and plotting methods on subsets of the data.

For instance, we may want to look for differences between countries *above* the median HDI, and countries *below* the median HDI.

We can use our Boolean filter as follows:

```{code-cell} ipython3
# Show countries with greater than median HDI.
above_median_HDI = df[hdi_gt_median]
above_median_HDI
```

We can "flip" the `True` and `False` values in the Boolean using the `~` operator (which we can read as "NOT").

This is the original Boolean Series:

```{code-cell} ipython3
# Show the Boolean Series
hdi_gt_median
```

We can reverse each `True` to a `False` and vice versa by placing the `~` symbol in front of the Boolean Series, like this:

```{code-cell} ipython3
# "Flip" the Boolean Series with the "~" operator
~hdi_gt_median
```

Let's use this operation (with `~`) to make a new Data Frame containing only the countries scoring *at or below* the median HDI:

```{code-cell} ipython3
# Use the `~` operator to show countries at or below median HDI
below_median_HDI = df[~hdi_gt_median]
below_median_HDI 
```

Now, on each Data Frame, we can call the `.describe()` method separately, to inspect the differences:

```{code-cell} ipython3
# Use the `.describe()` method with countries above the median HDI.
above_median_HDI['Fertility Rate'].describe()
```

```{code-cell} ipython3
# Use the `.describe()` method with countries at or below the median HDI.
below_median_HDI['Fertility Rate'].describe()
```

::: {exercise-start}
:label: below-replacement
:class: dropdown
:::

For a country to be at "replacement rate" - e.g. the rate at which the population will remain constant, rather than increase or decrease - the `Fertility Rate` must be above [2.1](https://ourworldindata.org/data-insights/which-countries-have-fertility-rates-above-or-below-the-replacement-level). 

Your task is to use a Boolean Series to calculate the proportion of countries in `df` which are below the 2.1 replacement rate.

Try to do this in as few lines as code as possible, using Pandas methods rather than functions from other libraries...

```{code-cell} ipython3
# Your code here
```

::: {exercise-end}
:::

::: {solution-start} below-replacement
:class: dropdown
:::

There are several ways to approach this, here are two.

First, for both methods we generate a Boolean Series using `df['Fertility Rate'] < 2.1` - this will be `True` where the `Fertility Rate` is *under* 2.1, and `False` otherwise.

In the first solution, we take the `sum()` of the `True` values, then divide that by the `len()` of the Data Frame (i.e. by the number of observations):

```{code-cell} ipython3
# Solution 1
below_replacement_bool_series = df['Fertility Rate'] < 2.1
below_replacement_bool_series.sum()/len(df)
```

The second solution is more efficient, as the `.mean()` method will perform the same calculation with less typing.

```{code-cell} ipython3
# Solution 2
below_replacement_bool_series.mean()
```

Strikingly, 80% of the countries in the Data Frame are below the replacement rate...

+++

::: {solution-end}
:::

+++

## Cleaning, summarizing and plotting data with Boolean Indexing

Remember, the data we have looked at on this page so far is just a fraction of the countries in the full HDI dataset. Let's import the full dataset, so we can use Boolean filtering to graphically inspect trends for countries above and below the median HDI. We will look at the data from all of the countries, but just for the year 2000, to keep the plot interpretable. 

We'll also use multiple methods from the [Pandas methods](0_2_pandas_dataframes_attributes_methods.md) page, to get the full data ready to generate these plots.

First, let's import the data:

```{code-cell} ipython3
# Import the full dataset.
full_df = pd.read_csv('data/children-per-woman-vs-human-development-index.csv')
full_df
```

This data is in *long format* - each row is an observation of one country, but each country appears in multiple rows. So, there are repeated observations from the same countries. If you look at the last few rows, you'll see repeated observations on Zimbabwe, over multiple years.

You'll also notice (an unfortunate commonality of many real datasets) that the full data contains many NaN values (we'll deal with some of these shortly).

As mentioned above, we will look at data from the year 2000, but using the full complement of countries. This will keep the plots interpretable. 

We can use Boolean filtering to strip the Data Frame down to just observations from the year 2000:

```{code-cell} ipython3
# Filter out all years apart from the year 2000
is_yr_2000 = full_df['Year'] == 2000  # Boolean Series.
full_df_2000 = full_df[is_yr_2000]  # Filter by indexing with Series.
full_df_2000
```

You can see that after this operation, the index labels (integer derived from the default `RangeIndex`) no longer correspond to row positions; there are 277 rows, but the largest row label is 59700.

Let's make a more usable index, but setting the `index` to be the three-letter country codes in the `Code` column:

```{code-cell} ipython3
# Set the index
full_df_2000 = full_df_2000.set_index('Code')
full_df_2000
```

You might notice that now we have `NaN` values in the index. We don't want this, so let's remove those observations.

We can again do this via Boolean filtering:

*NB:* the `~` operator, we are asking for rows with index labels that are NOT NaN...

```{code-cell} ipython3
# Create a Boolean Series from the Index using the `.isna()` method.
rows_are_na = full_df_2000.index.isna()
rows_are_na
```

```{code-cell} ipython3
# Remove NaN labeled rows by indexing with Boolean series.
full_df_2000 = full_df_2000[~rows_are_na]
full_df_2000
```

Let's inspect the index, to see what countries are left after removing the NaN values:

```{code-cell} ipython3
# Inspect the remaining values in the Index
list(full_df_2000.index)
```

If you scroll down this list, you can see that we have some rows that *maybe* do not correspond to countries (`OWID_WRL`, `OWID_KOS`, `OWID_SRM`).

Let's inspect these rows:

```{code-cell} ipython3
# Inspect rows with non-standard codes.
# Use direct indexing with list of column names to select columns.
full_df_2000.loc[['OWID_WRL', 'OWID_KOS', 'OWID_SRM']]
```

Ok, so we probably want to keep `Kosovo` and `Serbia and Montenegro`, but will drop the row with the data for the whole world.

We can use the `.drop()` method, with the `labels=` argument to tell Pandas to drop the row with the `'OWID_WRL'` label:

```{code-cell} ipython3
# Remove junk row with the `.drop()` method.
full_df_2000 = full_df_2000.drop(labels=['OWID_WRL'])
full_df_2000
```

We now have a clean Data Frame. Let's plot HDI against fertility rate, to graphically inspect the trend:

```{code-cell} ipython3
# Plot the data
full_df_2000.plot(x='Human Development Index',
                  y='Fertility Rate',
                  kind='scatter')
```

To plot the trend either side of the HDI median, let's create a Boolean Series based on the median HDI value.

This Series is `True` where the country has a greater than median HDI score, and `False` otherwise:

```{code-cell} ipython3
# Make a Boolean Series based on the median HDI
hdi_median = full_df_2000['Human Development Index'].median()
full_df_2000_gtr_than_median_HDI = full_df_2000['Human Development Index'] > hdi_median
full_df_2000_gtr_than_median_HDI
```

We can use this filter, and the `.plot()` method, to inspect the trend either side of the median HDI - (again, to get the values from the *opposite* side of the median, we can use the `~` operator):

```{code-cell} ipython3
# Plot countries above median HDI.
full_df_2000[full_df_2000_gtr_than_median_HDI].plot(x='Human Development Index',
                                                    y='Fertility Rate',
                                                    kind='scatter');
```

```{code-cell} ipython3
# Plot countries at or below median HDI.
full_df_2000[~full_df_2000_gtr_than_median_HDI].plot(x='Human Development Index',
                                                     y='Fertility Rate',
                                                     kind='scatter');
```

We can see graphically that the trends differ either side of the median, in a way that is clearer than looking at the full data on one plot. 

We can also use the same filtering procedure to view statistics on the subsets of countries either side of the median HDI:

```{code-cell} ipython3
# Getting `.describe()` stats for `Fertility Rate` from countries above the median HDI.
full_df_2000[full_df_2000_gtr_than_median_HDI]['Fertility Rate'].describe()
```

```{code-cell} ipython3
# Getting `.describe()` stats for `Fertility Rate` from countries below the median HDI (again, note the `~` operator)
full_df_2000[~full_df_2000_gtr_than_median_HDI]['Fertility Rate'].describe()
```

::: {exercise-start}
:label: a-big-pop
:class: dropdown
:::

Let's say we are interested in whether `Population` is associated with `Human Development Index`. We might inspect a scatter plot of the two variables, to visually inspect for a trend. However, there may be a problem:

![](images/big_pop_plot.png)

+++

The countries with huge populations are outliers which extend the x-axis far to the right, whilst most countries remain compressed near the far left of the x-axis. The pattern looks consistent with a random association, but perhaps there is a trend that is hard to see, in the presence of these extreme outliers?

Your task is to use Boolean filtering to "trim" out observations *above* the [25% percentile](https://en.wikipedia.org/wiki/Percentile) of the `Population` scores. You should then create a scatter plot with `Population` on the x-axis, and `Human Development Index` on the y-axis, from the trimmed data. This will allow you to inspect the trend of the cluster of countries with smaller populations.

You should try to do this in one or two lines of code, and using Pandas methods only.

```{code-cell} ipython3
# Your code here
```

::: {exercise-end}
:::

::: {solution-start} a-big-pop
:class: dropdown
:::

Our solution - to make it easier to read - creates this plot in two lines of code (though it is possible to do it one line, using method chaining). 

First, we use the "less than" (`<`) comparison operator to create a Boolean Series. This Series is `True` where the corresponding row in `full_df_2000` has a smaller-than-the-25th-percentile value, and `False` otherwise.

To calculate the value of the 25th percentile, we use `.describe()`. Astute readers may have noticed that `.describe()` reports the quartiles (25%, 50%, 75% percentiles) of numeric variables. Because `.describe()` returns a Series, we can use `.loc` indexing with the string `25%` to retrieve the value of the 25th percentile (the string `25%` is the `index` label of the 25th percentile, in the `.describe()` Series output). Another option here would be to use `df['Fertility Rate'].quantile(0.25)` to get the 25th percentile - there are always many roads to the same outcome in Pandas...

A second line of code then uses the `.plot()` method, with appropriate `x`, `y` and `kind` arguments, to generate the plot:

```{code-cell} ipython3
# Create a Boolean Series to filter the data
below_25_filter = full_df_2000['Population'] < full_df_2000['Population'].describe().loc["25%"]

# filter the data and plot
full_df_2000[below_25_filter].plot(x='Population',
                                   y='Human Development Index',
                                   kind="scatter");
```

Interesting, it looks like actually, there may be a negative linear trend in these countries, which was not visible in the scatterplot showing all of the data.

::: {solution-end}
:::

+++

::: {exercise-start}
:label: NaNs-army
:class: dropdown
:::

For this exercise, we want you to make the stuff of a data analyst's nightmares: the result of a filtering operation which contains only NaN values, and dashes (`'-'`) elsewhere. 

Do you remember the `full_df` Data Frame, from before we filtered it down to just rows corresponding to the year 2000?  Here it is:

```{code-cell} ipython3
# Show the full Data Frame.
full_df
```

This data contains many NaN values. Your task is to use Boolean filtering on this Data Frame, to return a Data Frame that contains:

* *only rows where `Fertility Rate` has NaN values...*
* ...and contains a dash character (`'-'`) for all other values *that are not NaN* (this should apply across all of the columns). 

The end product should look like this (and have the same dimensionality as the pictured Data Frame):

![](images/nan_army.jpg)

You can do this in one line of code, but it may be more readable to use several lines.

```{code-cell} ipython3
# Your code here
```

::: {exercise-end}
:::

::: {solution-start} NaNs-army
:class: dropdown
:::

As with the other solutions, there are multiple ways you could have done this. Here are two possible ways. Remember the `.isna()` method shown above? This creates a Boolean Series containing `True` values where data is NaN, and `False` values for non-NaN data. As such, you can get the "NaNs and dashes" Data Frame shown above by running the code in the cell below.

First, we filter the Data Frame to contain only rows where `Fertility Rate` is NaN. Then we use `.isna()` method to make the data across the whole Data Frame `True` where there are NaN values and `False` otherwise. We then use the `.replace()` method to replace `False` values with a dash string (`'-'`) and the `True` values with `np.nan`.

```{code-cell} ipython3
# Solution 1
full_df[full_df['Fertility Rate'].isna()].isna().replace(False, '-').replace(True, np.nan)
```

You could also use the Pandas *function* `pd.isna()` here, to get the same result...

```{code-cell} ipython3
# Solution 2
# We've used parentheses to split the single code line across multiple text lines.
(pd.isna(full_df[pd.isna(full_df['Fertility Rate'])])
 .replace(False, '-')
 .replace(True, np.nan))
```

::: {note}

**Upcoming changes to Pandas and `.replace`**

Note the cell at the top of this page with `pd.set_option('future.no_silent_downcasting', True)`.  If you did not run that cell, our solution would still work, but would give a rather confusing `FutureWarning:`, discussed in the link noted in that cell.

:::

We sincerely hope you never have to see another Data Frame like this one.

+++

::: {solution-end}
:::

+++

## Summary

This page has looked at the similarities/differences in filtering data with collections of Booleans in Numpy and Pandas.
