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

# Let There Be Data (Frames)

In [the](0_0_pandas_intro.Rmd) [previous](0_1_to_loc_or_iloc.Rmd)
[tutorials](0_2_pandas_dataframes_attributes_methods.Rmd) we showed how the
Pandas class objects (Series and Data Frames) are constructed from Numpy
objects (arrays) and other attributes.

We focused on the maxims:

- *"a Pandas Series is a numpy array, plus a `name` attribute and an array-like `index`"*

...and...

- "a Pandas DataFrame is just a *dictionary-like collection of Series*".

This page will look at several different ways of constructing Data Frames. All
of these use the `pd.DataFrame()` constructor but supply it with different
"ingredients". This influences the specific collection of attributes that the
resultant Data Frame will have.

```{code-cell} ipython3
# Import libraries
import numpy as np
import pandas as pd
```

## Reading in data from a file

The simplest, probably most common, and easiest way to create a Data Frame is to use a `pd.read_*` function to import data from a file.

`.csv` files are common way of storing data, and (as we have seen) can be imported using the creatively named `pd.read_csv()` function:

```{code-cell} ipython3
# Read in data the boring way
df_from_file = pd.read_csv('data/airline_passengers.csv')
df_from_file
```

Pandas, as a major Python data science library, has a large array of `read_*` functions, for importing data stored in different formats.

```{code-cell} ipython3
# Names in Pandas module starting with "read_"
[k for k in dir(pd) if k.startswith('read_')]
```

In other situations, and to deepen our understanding of Data Frame
construction, let's look at more elaborate, artisanal ways of creating Data
Frames...

+++

## Creating a blank Data Frame

Another very simple way to create a Data Frame is by using the
`pd.DataFrame()` constructor with no arguments:

```{code-cell} ipython3
# Calling the constructor with no arguments
df = pd.DataFrame()
df
```

Perhaps unsurprisingly, this returns a strange, blank output.

Again, unsurprisingly, many of the attributes of the Data Frame are also blank.

For instance, the index:

```{code-cell} ipython3
# Show the blank index
df.index
```

Ditto for the columns attribute:

```{code-cell} ipython3
# Show the blank columns.
df.columns
```

We can add new columns (e.g. new Pandas Series) into this blank Data Frame by using direct indexing on the left hand side (LHS). E.g.

```{code-cell} ipython3
# Create a new column in the Data Frame.
df['new_column'] = np.array([1, 2, 3])
df
```

We used a Numpy array to construct this new column, however, as we know, Data Frames are a dictionary-like collection of Series, so Pandas can represent the data as a Pandas Series:

```{code-cell} ipython3
# Show the type of df['new_column'].
new_col = df['new_column']
type(new_col)
```

The string which we used as the column name (e.g. `new_column`) has become the `name` attribute of this new Series:

```{code-cell} ipython3
# Show the `name` of the column.
new_col.name
```

...and the numpy array we supplied has become the `.values` of the Series:

```{code-cell} ipython3
# Show the `values` in the column.
new_col.values
```

Pandas has also automatically created a default `RangeIndex` for the Data Frame, because we did not specify what it should use as an index:

```{code-cell} ipython3
# Show the index of df.
df.index
```

As you saw in [The Pandas from Numpy page](0_0_pandas_intro), Series extracted from Data Frames inherit the `.index` of the Data Frame:

```{code-cell} ipython3
new_col.index
```

If we construct Data Frames using this method ("create a blank Data Frame, add the data later"), then any new columns we add must have equal numbers of elements.  This must be so, in order that the new column can share an index with the old.

```{code-cell} ipython3
# Add another new column with correct number of elements.
df['another_new_column'] = np.array(['A', 'B', 'C'])
df
```

If the number of elements differs, then Pandas will throw an error:

```{code-cell} ipython3
:tags: [raises-exception]

# ValueError from wrong number of elements on RHS.
df['a_further_new_column'] = np.array([4, 5 , 6, 7])
```

Notice the text of this error: `ValueError: Length of values (4) does not match length of index (3)`. The error is caused because all columns must share an index, to facilitate the label-based indexing (via `.loc`) that we have seen on previous pages.

We want to avoid the [pitfalls](0_1_to_loc_or_iloc.Rmd) of integer indices, such as `RangeIndex` (e.g. misalignment between the integer location of data, and the numerical index label of that data). To do this, we can specify a non-integer values for the index, after we have created the Data Frame.

```{code-cell} ipython3
# Set the index
df.index = ['Person_1', 'Person_2', 'Person_3']
df
```

We can also specify the index directly when we make the "blank" Data Frame:

```{code-cell} ipython3
df_again = pd.DataFrame(index=['Person_1', 'Person_2', 'Person_3'])
df_again
```

This creates a Data Frame with only an index, which data can then be added to:

```{code-cell} ipython3
df_again['new_column'] = np.array([1, 2, 3])
df_again
```

Because all Series/columns in the Data Frame must share an `index`, Pandas will predictably throw an error if try to use something that is the wrong length/shape to be a valid `index`:

```{code-cell} ipython3
:tags: [raises-exception]

# ValueError because we have specified the wrong number of index elements.
df.index = ['Person_1', 'Person_2', 'Person_3', 'Person_4']
```

Again, the error that Pandas gives us here is informative: `ValueError: Length mismatch: Expected axis has 3 elements, new values have 4 elements`. (Unfortunately, not all Pandas errors are as obvious as this one).

+++

## Constructing a Data Frame from a dictionary of Numpy arrays

Another common way to construct Data Frames is to use a dictionary.

When we do this, the keys of the dictionary become the column names (and therefore the `name` attribute of the Series that constitutes a given column); and the values of the dictionary become the `values` attribute of a given column.

First, let's make a dictionary:

```{code-cell} ipython3
# Make a dictionary, using the keys "A" and "B" and two Numpy arrays for the values
dictionary = {'A': np.array([1, 2, 3, 4]),
              'B': np.array([5, 6, 7, 8])}
dictionary
```

Here are the keys and values of the dictionary, containing this toy data:

```{code-cell} ipython3
# Show the keys of the dictionary
dictionary.keys()
```

```{code-cell} ipython3
# Show the values of the dictionary
dictionary.values()
```

We can pass this dictionary to the `pd.DataFrame()` constructor. As noted above, the keys will become the `name` attribute of each column (where each column is a Pandas Series). The values will become the `.values` attribute of each column:

```{code-cell} ipython3
# Construction from a dictionary
df3 = pd.DataFrame(dictionary)
df3
```

As we know, the Data Frame itself is just a dictionary-like collection of Series:

```{code-cell} ipython3
# Show one column/Series
df3['A']
```

Each Series inherits its `name` attribute from its *key* in the original dictionary:

```{code-cell} ipython3
df3['A'].name
```

...and its `.values` attribute from the *values* in the original dictionary:

```{code-cell} ipython3
df3['A'].values
```

## Constructing a Data Frame from a dictionary of Pandas series

We can also use Pandas Series as the values in a dictionary (rather than Numpy
arrays), in order to build a Data Frame. Because Pandas Series contain a Numpy
array plus additional attributes, like an `index`, we need to be aware of this
when using them to create Data Frames, as conflicts between the indexes of
different Series can lead to errors.

Let's build a Series with the familiar three-letter country codes, the country names, and the [HDI data](https://ourworldindata.org/grapher/children-per-woman-vs-human-development-index):

```{code-cell} ipython3
# Make an array containing the country codes
country_codes_array = np.array(['AUS', 'BRA', 'CAN',
                                'CHN', 'DEU', 'ESP',
                                'FRA', 'GBR', 'IND',
                                'ITA', 'JPN', 'KOR',
                                'MEX', 'RUS', 'USA'])
```

```{code-cell} ipython3
# Make an array containing the country names
country_names_array = np.array(['Australia', 'Brazil', 'Canada',
                                'China', 'Germany', 'Spain',
                                'France', 'United Kingdom', 'India',
                                'Italy', 'Japan', 'South Korea',
                                'Mexico', 'Russia', 'United States'])
```

As previously, we will use the country codes as an `index`:

```{code-cell} ipython3
# Build a Series of the country names
country_names_series = pd.Series(country_names_array,
                                index=country_codes_array)
country_names_series
```

Now, let's do the same for the HDI scores:

```{code-cell} ipython3
# Human Development Index Scores for each country
hdis_array = np.array([0.896, 0.668, 0.89 , 0.586,
                       0.844, 0.89 , 0.49 , 0.842,
                       0.883, 0.709, 0.733, 0.824,
                       0.828, 0.863, 0.894])
```

Here also we will use the country codes as the index:

```{code-cell} ipython3
hdi_series = pd.Series(hdis_array, index=country_codes_array)
hdi_series
```

We can then create the Data Frame by using the Series as values in a dictionary, and passing that dictionary to the `pd.DataFrame()` constructor:

```{code-cell} ipython3
df4 = pd.DataFrame({'country_names': country_names_series,
                    'HDI': hdi_series})
df4
```

However, it is very important when using this method to ensure that all the
Series share an index.

Strange things can happen if they do not.

Let's adjust the `hdi_series` to give it a numerical index:

```{code-cell} ipython3
# Adjust the `hdi_series` to have a numerical index
# Copy the Series with the Series `.copy` method.
hdi_with_int_index = hdi_series.copy()
hdi_with_int_index.index = np.arange(len(hdi_series))
hdi_with_int_index
```

For the latest Pandas (2.2.3 at time of writing), Pandas will give an error if
we try to construct a Data Frame from a dictionary with these two Series as
the values:

```{code-cell} ipython3
:tags: [raises-exception]

# TypeError if we construct a Data Frame using Series without matching indexes
df5 = pd.DataFrame({'country_names': country_names_series,
                    'HDI': hdi_with_int_index})
```

::: {exercise-start}
:label: incompatible-series
:class: dropdown
:::

In the cell above, at least at time of writing, you get the following error:

```
TypeError: '<' not supported between instances of 'int' and 'str'
```

This occurred when we passed one Series with `int`-type Index values, and
another with `str`-type Index values.

Reflect back on the [first exercise](differing-indices) in the [Pandas from
Numpy page](0_0_pandas_intro).  Why do you think Pandas is comparing `int`s to
`str`s as it creates the Data Frame?

::: {exercise-end}
:::

::: {solution-start} incompatible-series
:class: dropdown
:::

Working through the [indices exercise](differing-indices) should have revealed
that Pandas follows something like the following algorithm, when dealing with
the `.index` of different Series intended for a Data Frame:

* First check if the Series Indices are the same.  If so, use the Index of any
  Series.
* If they are not the same, first sort all Series by their Index values, and
  use the resulting sorted Index.

The first of these two steps will involve comparing `int` to `str`, hence the error.

::: {solution-end}
:::

+++

Remember each index label is a identifier for each row of the Data Frame. Pandas is trying to compare the indices of the two series in order to match corresponding rows, and failing, because it cannot compare the string index of `country_names_series` to the (newly set) integer series of `hdi_series`.

[Later on](0_4_dealing_with_missing_data_in_pandas.Rmd) we will see further signs that Pandas is trying to match rows between series by using the `index`.

+++

## Constructing a Data Frame from a single Pandas series

`pd.DataFrame` has a special case in which you pass a single Series as the data argument.

```{code-cell} ipython3
df_single = pd.DataFrame(hdi_series)
df_single
```

**Be careful** - as you will see below, if you pass a *sequence* of Series, then the Series become the *rows*.  Here, the single Series becomes a single column in the Data Frame.

The column name comes from the Series name:

```{code-cell} ipython3
hdi_series.name
```

As you remember, Series have an optional `.name` (for which the default is `None`).  For example:

```{code-cell} ipython3
hdi_series_no_name = pd.Series(hdis_array, index=country_codes_array)
hdi_series_no_name.name is None
```

If you pass a Series with no `.name` (`.name == None`) then Panda must make a default column name.  It uses the same default for column names as it does for row names, that is, a `RangeIndex` containing integers, where, in this case, it only contains the integer value `0`:

```{code-cell} ipython3
df_single_no_name = pd.DataFrame(hdi_series_no_name)
df_single_no_name
```

```{code-cell} ipython3
df_single_no_name.columns
```

Indexing for this column, with an integer label, is likely to become confusing:

```{code-cell} ipython3
# Getting the column by label.
df_single_no_name.loc[:, 0]
```

Or even this (which is very confusing - direct indexing with column name):

```{code-cell} ipython3
# Direct indexing using column name, where name is integer 0
df_single_no_name[0]
```

It's usually advisable to either - set the Series name when constructing the Series, or later, with (e.g.) `hdi_series.name = 'Human Development Index'` - or set the name explicitly to `pd.DataFrame` using the `columns=` argument:

```{code-cell} ipython3
# Setting the column name or names on constructing the Data Frame.
df_single_now_named = pd.DataFrame(hdi_series_no_name,
                                   columns=['My HDI'])
df_single_now_named
```

## Constructing a Data Frame from a sequence of Pandas series

+++

Series have an optional `.name` (for which the default is `None`).

If we specify a `.name` for each Series, then we can pass a sequence of these named Series to `pd.DataFrame`; Pandas interprets these Series as *rows* in the Data Frame.  For example:

```{code-cell} ipython3
# Set not-default names for the Series.
country_names_series.name = 'country_names'
hdi_series.name = 'HDI'
df5 = pd.DataFrame([country_names_series, hdi_series])
df5
```

Notice the `.names` of the Series become the `.index` values of the Data Frame
(the row labels).  The .`index` of the two Series become the column labels. To
get the same effect as we have had, up until now, we can *transpose* the Data
Frame, so that the rows become columns, and the columns become the rows:

```{code-cell} ipython3
# .T is the transpose attribute of the Data Frame.  It returns a new, transposed Data Frame.
df6 = df5.T
df6
```

<!--- Fuse with stuff from Strange and nameless Data Frame columns in 0_1_ ... -->

+++

## Summary

This page has looked at different methods of constructing Data Frames, and how these affect different attributes of the Pandas Series that constitute each Data Frame.
