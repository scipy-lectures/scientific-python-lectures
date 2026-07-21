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

# More Pandas methods

<!---
📝 NOTE: Covered here:

* string operations, ex: df[‘name’].str[2:4], df[‘name’].str.count('a') [For
  this, compare different country names between Mat > Mort data and
  Dev/Fertility data, e.g. different names for Russia and South Korea])

* Strings (and dates) have a special accessor to perform vectorized string (or
  date) operations

* specialized accessor methods Series.str

**Matthew Notes: do everything on a Series first. Series.str follow the pandas
from numpy structure**

**For any string operations, show different routes to the same outcome e.g. via Boolean indexing and then replace etc.**

Structure of this page:

* show python string methods/string methods with numpy arrays

* build a Series and show Series.str methods

* show str methods with a dataframe + how this can be very useful when
  cleaning data
-->

String/text data is often used to represent categorical variables, and commonly appears in a variety of data analysis contexts. When dealing with string/text data we will frequently find that we need to alter the strings to correct errors, improve clarity, make formatting uniform, or for a host of other reasons.

String methods are inherent to Python, and these methods or variants of them can all be used on Numpy arrays and Pandas Series/Data Frames. However, Numpy and Pandas use different interfaces for interacting with strings. To understand the differences between Numpy and Pandas with respect to strings, let's begin at the foundation, the in-built string methods of Python. The cell below contains a simple string.

*NB*: if you are running this tutorial interactively, and you like messy cell outputs, you may want to use `dir()` again here to see the full complement of available string methods.

```{code-cell} ipython3
# A string
string = "string_of_words"
string
```

Remember that *a method is a function attached to an object*. In this case our object is a string. 

Let's say we like reading values in our data as if they are being spoken in a loud voice. If this is the case we can alter the format of the string to make all letters uppercase, using the `.upper()` method:

```{code-cell} ipython3
# The `.upper()` method of `str`.
string.upper()
```

We can replace characters in the string using the aptly named `.replace()` method. Here we supply two strings to the method, first the string we want to replace, and then second, the string we want to replace it with. In this case, let's `.replace()` the underscores with a blank space:

```{code-cell} ipython3
# The `.replace()` method
string.replace('_', ' ')
```

Fancier formatting methods will let us adjust strings, for instance, in title case (`.title()`):

```{code-cell} ipython3
# A more elaborate string method
string.title()
```

In Python, strings are collections of characters, and so we can slice them as we would a list or array, using integer indexes and the `:` symbol:

```{code-cell} ipython3
# Slicing with strings
string[0:1]
```

```{code-cell} ipython3
string[0:2]
```

```{code-cell} ipython3
string[0:7]
```

You can visit [this page](https://www.w3schools.com/python/python_ref_string.asp) to see the variety of string methods available in base Python.

+++

## String methods with Numpy arrays

So, strings in base Python have a large number of in-built methods - what about strings in Numpy?

Numpy arrays themselves do not have specific string methods, but the in-built Python string methods can be called on individual string values in a Numpy array. Alternatively, we can use functions from the `np.char.` module to operate on all the strings in the array concurrently.

To investigate how string data is handled in Numpy, let's make some arrays containing strings from now (very) familiar [HDI dataset](https://ourworldindata.org/grapher/children-per-woman-vs-human-development-index):

```{code-cell} ipython3
# Import libraries (no imports were needed prior to this point as string methods are part of base python)
import numpy as np
import pandas as pd
```

```{code-cell} ipython3
# A custom function to generate a Series to check exercise solutions.
import clean_gender_df_names
```

```{code-cell} ipython3
# Calculate answer to exercise (see below).
answer_clean_series = clean_gender_df_names.get_cleaned(
    pd.read_csv("data/gender_stats.csv")['country_name'])
```

```{code-cell} ipython3
# Standard three-letter code for each country.
country_codes_array = np.array(['AUS', 'BRA', 'CAN',
                                'CHN', 'DEU', 'ESP',
                                'FRA', 'GBR', 'IND',
                                'ITA', 'JPN', 'KOR',
                                'MEX', 'RUS', 'USA'])
```

```{code-cell} ipython3
# Country names.
country_names_array = np.array(['Australia', 'Brazil', 'Canada',
                                'China', 'Germany', 'Spain',
                                'France', 'United Kingdom', 'India',
                                'Italy', 'Japan', 'South Korea',
                                'Mexico', 'Russia', 'United States'])
```

For comparison, let's make an array containing the numerical HDI scores:

```{code-cell} ipython3
# Human Development Index Scores for each country
hdis_array = np.array([0.896, 0.668, 0.89 , 0.586, 
                       0.844, 0.89 , 0.49 , 0.842, 
                       0.883, 0.709, 0.733, 0.824,
                       0.828, 0.863, 0.894])
```

The `dtype` attribute of the first two arrays begins with `<U`, indicating we are dealing with string data.

```{code-cell} ipython3
# Show the dtype of the country codes array (e.g. string data)
country_codes_array.dtype
```

`U3` tells us that the array stored Unicode (`U`) strings up  three Unicode characters in length.

```{code-cell} ipython3
# Show the dtype of the country names array (e.g. string data)
country_names_array.dtype
```

Conversely, the `hdis_array` contains data of a numerical type:

```{code-cell} ipython3
# Show the dtype of the hdis array (e.g. numeric data)
hdis_array.dtype
```

Using indexing, we can use all of the in-built Python string methods on the individual values within a Numpy array:

```{code-cell} ipython3
# Methods on an individual string
country_codes_array[0]
```

For instance, we can change the case of the value:

```{code-cell} ipython3
# Lowercase
country_codes_array[0].lower()
```

```{code-cell} ipython3
# Uppercase
country_codes_array[0].upper()
```

We can also replace elements of the string:

```{code-cell} ipython3
country_codes_array[0].replace("A", "Comparable to the ")
```

Understandably, if we try to use any of these string methods on numerical data, we will get an error:

```{code-cell} ipython3
:tags: [raises-exception]

# Oh no!
hdis_array[0].upper()
```

All of the string methods used in this section above have been called on single string values from a Numpy array. If we try to use a string method on all values of the array simultaneously, we will also get an error:

```{code-cell} ipython3
:tags: [raises-exception]

# This does not work
country_codes_array.lower()
```

String methods in Numpy must be called from the single string values or using the `.char.` module.

For example, we can use the `np.char.lower()` function to operate on all values of the Numpy array at once:

```{code-cell} ipython3
# This DOES work
np.char.lower(country_codes_array)
```

```{code-cell} ipython3
# This DOES work too
np.char.replace(country_codes_array, 'A', '!')
```

Pandas deals with string data slightly differently to Numpy. The elements of the `.values` component of a Pandas Series can be operated on altogether by using the `.str.` accessor, to which we will now turn our attention.

+++

## String methods with Pandas Series

As mentioned above, Pandas Series have a specialised accessor (`.str.`) which bypasses the need to use `np.char.` when we want to do something to all of the string values in a Series.

To see how this works, let's construct a Series from our `country_names` array:

```{code-cell} ipython3
# Show again from Series
names_series =  pd.Series(country_names_array,
                          index=country_codes_array)
names_series
```

To use the `.str.` accessor, we just place it after our object (e.g. our Series containing our string data). We then can call a variety of string methods, which we be applied to all elements in the `.values` array of the Series:

```{code-cell} ipython3
# String methods on Series
names_series.str.upper()
```

```{code-cell} ipython3
# The `.str.lower()` method
names_series.str.lower()
```

The `.replace()` string method is also available here, it will operate on all the elements in the Series, though in this case (as there is only one `United States`) it will only alter one value:

```{code-cell} ipython3
# Replacing values in the Series
names_series.str.replace("United States", "USA")
```

By default, the **string-specific** replace method (accessed through the `str` accessor - `.str.replace()`) will search for expressions *within* strings, as opposed to searching for exact, whole string matches. 

This is different to the behaviour of the **non-string-specific** `.replace()` method which we encountered on an [earlier page](0_2_pandas_dataframes_attributes_methods). By default the non-string-specific `.replace()` method will search for *exactly matching whole strings*. Confusion between these two methods is a [commong source of error](https://stackoverflow.com/a/50614413/23148902). 

As such, by using the string-specific `.str.replace()` method we can easily replace substrings in multiple elements in the data at once, even where the whole strings are not the same. For instance:

```{code-cell} ipython3
# Replacing values in the Series
names_series.str.replace("United", "Unknown")
```

Here, the substring `United` has been replaced with `Unknown` in the both the strings `"United Kingdom"` and `"United States"`. This is a good example of a situation where a specialized accessor (like `.str.`) can change the behaviour of other methods, often in a helpful way for dealing with a specific data type.

The syntax for slicing strings is the same as for a single value, but it also operates across all elements in the Series at once:

```{code-cell} ipython3
# Slicing with strings, in a Series
names_series.str[2:4]
```

Using the `.contains()` method, Boolean Series can be generated by searching for specific instances of a substring in each value:

```{code-cell} ipython3
# Generate a Boolean Series, True where the value contains "Ind"
names_series.str.contains("Ind")
```

These Boolean Series can be used to retrieve specific values from the original Series, via Boolean filtering:

```{code-cell} ipython3
# Use Boolean filtering to retrieve a specific datapoint
names_series[names_series.str.contains("Ind")]
```

## String methods with Pandas DataFrames

So, Pandas makes it somewhat easier than Numpy to perform operations on all the string elements at once. 

Remember that *a DataFrame is a dictionary-like collection of Series*, and so everything we have just seen of strings in Pandas Series applies to the *columns* of a Data Frame.

Let's import the [HDI data](https://ourworldindata.org/grapher/children-per-woman-vs-human-development-index) in a Pandas Data Frame:

```{code-cell} ipython3
# Import data
df = pd.read_csv("data/year_2000_hdi_fert.csv")
# Show the data
df
```

Because each Data Frame column can be extracted as a Pandas Series, we can use string methods in the same way as we saw in the last section:

```{code-cell} ipython3
# Use the `.replace()` method
df['Country Name'].str.replace('I', 'I starts I')
```

However, Pandas will not let us use string methods on the whole Data Frame at once:

```{code-cell} ipython3
:tags: [raises-exception]

# Cannot use `.str` methods on whole Data Frame.
df.str.upper()
```

We might think that this is because not all the data in the Data Frame is of
the string type, but this is not the case. We also cannot use Pandas string
methods on Data Frames with columns only containing string data, as we get the
same error:

```{code-cell} ipython3
:tags: [raises-exception]

# Oops, using `.str` fails even for all-string-dtype columns.
df[['Country Name', 'Code']].str.len()
```

Notice here we did not set `Code` as the index of the Data Frame, we are just treating it as a column containing string data. The cell below sets it as the index, so we can use label-based indexing later in this tutorial:

```{code-cell} ipython3
# Set the index
df.index = df['Code']
```

So, we cannot apply string methods to multiple columns at once, but if we focus on one column, we can use all of the available string methods:

```{code-cell} ipython3
# The `.lower()` method
df['Country Name'].str.lower()
```

```{code-cell} ipython3
# The `.upper()` method
df['Country Name'].str.upper()
```

```{code-cell} ipython3
# Using the `str.count()` method
df['Country Name'].str.count('a') 
```

```{code-cell} ipython3
# The `str.contains()` method
df['Country Name'].str.contains('Russia')
```

```{code-cell} ipython3
# See if there are any Trues in the Series
df['Country Name'].str.contains('Russia').sum()
```

```{code-cell} ipython3
# Filtering data using the `str.contains()` method
df[df['Country Name'].str.contains('Russia')]
```

## Uses of string methods in data wrangling

As we mentioned earlier, string methods generally useful for cleaning text data. This can be especially useful when combining data from different sources, where different conventions in data entry may lead to similar data being formatted differently.

To explore this, let's import a new dataset, which, like the HDI data, contains observations at the country level (e.g. each row is an observarion from a specific country).

This dataset is also at the country-level of granularity, and it contains various data about countries, including maternal mortality rates. You can read more about the dataset [here](data/gender_stats).

```{code-cell} ipython3
# Import gender statistics dataset
gender_df = pd.read_csv("data/gender_stats.csv")
gender_df
```

```{code-cell} ipython3
gender_df[gender_df['country_name'].str.startswith('S')]
```

Let's say we are interested in Russia, but do not know how the name of the country is formatted. We can use the `str.contains()` method to search for likely matches.

```{code-cell} ipython3
# Hmmm is Russia not in this data?
gender_df['country_name'].str.contains('Russia')
```

That output is pretty opaque, maybe there is a `True` in there somewhere. Because Python treats `True` values as being equal to `1`, we can chain on the `.sum()` method to count the number of `True` values in the above Boolean Series:

```{code-cell} ipython3
# Count the Trues for country names containing "Russia" in the `maternal
gender_df['country_name'].str.contains('Russia').sum()
```

It appears we do have one match. Let's use the Boolean Series we just made to have a look at the row that contains the string "Russia" in the `country_name` column:

```{code-cell} ipython3
# Use the `str.contains()` method to filter the data
gender_df[gender_df['country_name'].str.contains('Russia')]
```

So, we have found the row for Russia in this new dataset. Let's compare the naming convention to the HDI data, in the `df` Data Frame:

```{code-cell} ipython3
# Get the data for Russia, from the HDI data
df.loc['RUS']
```

In due course, we may want to
[merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)
these datasets. To do that, we need common identifiers linking rows in each
dataset which refer to the same observational units (in this case countries).

String methods are our friend here. We can use the process just outlined for
find data for a specific country, and then use other methods to ensure uniform
formatting between the datasets, such that we can merge them:

```{code-cell} ipython3
# Format the maternal mortality data for Russia to use the same country name as the HDI data
gender_df['country_name'] = gender_df['country_name'].str.replace('Russian Federation', 'Russia')

# Show the newly formatted row
gender_df[gender_df['country_name'].str.contains('Russia')]
```

We are now ready for a clean and stress-free data merge! (*NB*: we are grossly exaggerating here, merging datasets is almost never stress-free...)

+++

::: {exercise-start}
:label: strings-and-things
:class: dropdown
:::

The `gender_df['country_name']` Series contains a lot of formatting that is nice to read, but annoying
to use in indexing operations (or any time where we need to type them).

Entries like `'Virgin Islands (U.S.)'` and `'St. Martin (French part)'` will be a pain to type if we need to use them in `.loc` indexing operations, for instance.

We would therefore like to create a new Series containing versions of these names that are easier to type.

That is what we have done with some hidden code.   The hidden code:

* Processes the `gender_df['country_name']` Series to make a new Series where we have replaced the original names (above) with versions of these names that are easier to type.
* Taken this new Series, and run `sorted(new_series.unique())` to show you the new names.

Have a careful look at the resulting list below - and work out which Pandas string methods have been used to get from the `gender_df['country_name']` Series to the new Series, to which we have applied `sorted(new_series.unique())`

```{code-cell} ipython3
# This is the answer - don't use it in your solution.
sorted(answer_clean_series.unique())
```

Your task now is to make a Series called `my_clean_series` which gives (with `sorted(my_clean_series.unique())`) a list that is *identical* to the list shown above.

You can perform the relevant string transformations using Pandas string methods on the `gender_df['country_name']` Series, and then run `sorted(my_clean_names.unique())` to get the final array.

There is a cell at the end of the exercise to check your answer.

Try to do the string transformation in as few lines of code as possible and **using ONLY Pandas string methods**.

**Hint**: There are many ways to do this, but for maximum beauty, you might consider having a look at [Python's str.maketrans function](https://docs.python.org/3.3/library/stdtypes.html?highlight=maketrans#str.maketrans).  And yes, you can use `str.maketrans` as well.  Or you can use some other algorithm of your choice.

```{code-cell} ipython3
# Your code here to create a new Pandas Series with modified
# country names, as above.
my_clean_series = pd.Series()  # Edit here to solve the problem.
# ...
# But don't modify the code below.
my_clean_names = sorted(my_clean_series.unique())
my_clean_names
```

```{code-cell} ipython3
# Run this cell to check your answer.
# It will print 'Success' if your cleaning worked correctly.

def check_names():
    answer_list = sorted(answer_clean_series.unique())
    if len(my_clean_names) != len(answer_list):
        return 'The answers are of different lengths'
    not_matching = np.array(my_clean_names) != answer_list
    if not_matching.any():
        print('My solution unmatched', my_clean_names[not_matching])
        print('Desired unmatched', cleaned_names_answer[not_matching])
        return 'Remaining unmatched values'
    return 'Success'
    
check_names()
```

::: {exercise-end}
:::

::: {solution-start} strings-and-things
:class: dropdown
:::

Our solution is below. We have used the hint above to make a *translation table* from a set of characters to another set of characters, followed by a set of characters to delete, and then applied this translation table with the Pandas `.str.translate` method.

We then use the `.lower()` method to remove the capitalization.

To make the array *identical* to the one shown above (and the one used for marking this exercise), we then re-rename `russia` to `russian_federation` before using `.unique()` to show the final array:

```{code-cell} ipython3
soln_clean_series = (gender_df['country_name']
                     .str.lower()
                     .str.translate(str.maketrans(' -', '__', '().,'))
                     .str.replace('russia', 'russian_federation'))
soln_clean_names = sorted(soln_clean_series.unique())
soln_clean_names
```

::: {solution-end}
:::

+++

## Summary

This page looked at string methods in base python, Numpy and Pandas.

Numpy and Python inherit their string methods from base python, but apply them in different ways. 

Numpy does not have a set of methods for applying string methods to every element of an array simultaneously. We need functions from the `np.char` module if we want this.

By contrast, Pandas Series - whether in isolation or as columns in a Data Frame - have the `.str.` accessor for easily performing string operations on every element in a Series.
