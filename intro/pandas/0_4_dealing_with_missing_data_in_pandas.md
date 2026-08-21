---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Handling data that is missing

<!---
📝 NOTE:
Covered on this page/structure of this page:

* Make an array and series containing NaNs and show how pandas and numpy treat them differently

* Populating the region column? E.g. replacing the missing data?

``
# drop NaNs gives the same mean
df['Fertility Rate'].loc['ZWE'].dropna().mean()
``

* NaNs are interpreted as missing data and ignored in most operations
* Pandas uses NaN as a flag, not as an indication of a failed floating point
  operation
* NumPy does not have a concept of a missing value, NaNs propagate.
* In Pandas NaN is an indication of missing data - Pandas will by default
  drop nans from most operations).s
-->

[NaN](https://en.wikipedia.org/wiki/IEEE_754) is short for Not-a-Number.

This page will compare how NaN values differ between NumPy and Pandas.

You are probably aware that NumPy will produce NaNs from invalid floating
point operations, such as dividing 0 by 0. In Pandas, NaNs are more commonly
a flag to indicate the absence of data, for floating point and other data
types.

We will also look at how to handle NaNs safely in Pandas. First, let's remind ourselves how NaNs work in NumPy.

+++

## NaNs in NumPy

As mentioned above, NaNs in NumPy result from invalid floating point
operations.

```{code-cell}
# Import libraries
import numpy as np
import pandas as pd

# NaN results from a NumPy operation dividing 0 by 0
a_nan = np.array(0) / np.array(0)
a_nan
```

There are potential pitfalls when dealing with NaN values, to which we will
now turn our attention.

+++

As you see above, the NumPy `dtype` of the returned NaN value is `float64`.
This tells us that the NaN value is a special and particular type of floating
point value, in the same sense that Inf (infinity) or -Inf (negative
infinity) are special floating point values:

```{code-cell}
# Inf (np.inf) is another special floating point value.
np.array(1) / np.array(0)
```

NumPy uses this special NaN (`np.nan`) value to indicate that the value is
_invalid_. We will soon see that Pandas uses `np.nan` in a different and expanded meaning. But more of that in a little while.

The logic of NaNs as _invalid values_ means that _any_ operation with a NaN should return — a NaN — because any operation with an invalid value must itself be an invalid value. This propagation can have some superficially unexpected consequences that can trap the unwary:

```{code-cell}
# A (potentially) unexpected False
a_nan == np.nan
```

```{code-cell}
# Another strange result with the equality operator
np.nan == np.nan
```

The last two cells above both return a `False` value because NaN value is
treated as an invalid value. For a NaN, a value is _invalid_, so
comparing any value to a NaN is itself a NaN, even if the other value is
a NaN.

To ask the question _is this value a NaN_, use `np.isnan()`:

```{code-cell}
np.isnan(a_nan)
```

The same principles apply when we are dealing with NaN values in an array:

```{code-cell}
# A new array with NaN and non-NaN values
arr = np.array([np.nan, np.nan, 1, 3])
arr
```

Again, if we want to find which of these values as NaNs, we might (early in our programming careers) try something like this:

```{code-cell}
# Probably not what you meant.
arr == np.nan
```

This has failed to identify the NaN elements, because NaNs propagate, and therefore any operation with a NaN gives a NaN, and therefore the comparison `==` with a NaN value returns NaN.

You may well want `np.isnan()` here; it does ask the question — which of these values are NaNs — returning True where the value is NaN, and False otherwise.

```{code-cell}
# Probably what you did mean.
np.isnan(arr)
```

Perhaps more obviously, any mathematical operation with NaN gives NaN:

```{code-cell}
# Multiplying NaNs by something.
arr * 2
```

```{code-cell}
# Adding with NaNs
arr + 2
```

OK, so the [TL;DR](https://en.wikipedia.org/wiki/TL;DR) here is that in NumPy
NaNs signal an invalid operation has taken place.

_NaNs propagate_; any numerical operation a NaN will result in a NaN.

Let's compare this to the way that Pandas uses NaNs.

+++

## NaNs in Pandas

We have seen that NaN values in NumPy are values that indicate the result of
invalid floating point operations.

The function of NaN values in Pandas is somewhat different.

NaN values in Pandas are _flags for missing data_. The NaN values themselves
possess the same properties and pitfalls that we showed in the last section.
However the _cause_ of NaNs in Pandas is most often that _data was missing_
rather than invalid numerical operations being performed on the data.

We'll say more about what Pandas means by _missing_ below.

Let's explore these concepts further by importing a dataset. We will use the
full version of the [Human Development
Index](https://ourworldindata.org/grapher/children-per-woman-vs-human-development-index)
dataset, which contains values for every year, rather than just the subset of
data from the year 2000, which we have looked at on previous pages:

```{code-cell}
# Import data as Data Frame.
df = pd.read_csv("data/children-per-woman-vs-human-development-index.csv")
# Set the index to the country codes.
df = df.set_index('Code')
df
```

It is immediately apparent that this dataset contains more NaNs than
a retirement village.[^also-nans]

[^also-nans]:
    We apologise to our readers outside the US or UK, but this was
    a small pun, because "nan" is a fairly popular way to refer to your
    grandmother.

Look at the `Human Development Index` column (which we extract as a Series):

```{code-cell}
# A column with lots of NaN values.
df['Human Development Index']
```

Let's take a closer look at the value in the first row of this column:

```{code-cell}
# Show a nan value
df['Human Development Index'].iloc[0]
```

Sure enough, it's a NaN. It also has the expected `float64` data type that we
saw above.

+++

We can use `np.isnan()` to get a Boolean confirmation that we are in the
presence of a standard NaN:

```{code-cell}
# Using the `np.isnan()` function
np.isnan(df['Human Development Index'].iloc[0])
```

Alternatively, you could use Pandas' own `pd.isna()` function:

```{code-cell}
# Using the `np.isnan()` function
pd.isna(df['Human Development Index'].iloc[0])
```

::: {note}

**Pandas NA**

In fact `pd.isna()` doesn't just check for NaN values, because Pandas has some
other, less common ways to indicate an element is missing. You will see this
hinted in the name `pd.isna()`, because Pandas thinks of missing values as
`NA` values, where NA [seems to stand for Not-Applicable or
Not-Available](https://stats.stackexchange.com/questions/72907/in-statistics-what-does-na-stand-for).
See [Working with Missing
Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
for the full gory details. The summary at this stage is:

- For readability, and to allow for the use of other NA values in Pandas, we
  suggest you use `pd.isna()` or Pandas `.isna()` methods to check for missing
  values.
- That said, at the moment, Pandas nearly always indicates missing (NA) values
  with NumPy's `np.nan`.

:::

Missing data - fancifully referred to as _missingness_ - is common in the vast
majority of datasets encountered in the wild. _Missing data_ means data that,
for whatever reason, are not present for a particular row and column.

Above we saw, for example, that the HDI value for Afghanistan, and 1950, is missing (not available).

```{code-cell}
df.iloc[0]
```

That is probably because there were no relevant statistics, for Afghanistan,
in 1950, with which to calculate the value.

+++

We will nearly always want to know _how much_ of a given dataset is missing, as we will need to factor this in as a limitation of our data analysis.

::: {exercise-start}
:label: missing-mysteries
:class: dropdown
:::

Why might we worry about missing values? Why can't we just drop them and forget about them? Let's load some [related data from the World Bank](data/gender_stats) with country statistics on various measures:

```{code-cell}
gender_df = pd.read_csv('data/gender_stats.csv')
gender_df
```

Notice that there are various NaN values here.

A) Do you think these indicate invalid floating point operations at some previous step, or do they indicate missing (Not Available) data? Why?

+++

**Write your answer here, replacing this text.**

+++

Here is a calculation of the mean Health Exp(enditure) per Cap(ita) (per person):

```{code-cell}
gender_df['health_exp_per_cap'].mean()
```

Do you think this value is a reasonable estimate of actual worldwide health expenditure per person? If not, why not? Can you think of any way of improving this estimate?

+++

**Write your answer here, replacing this text.**

::: {exercise-end}
:::

::: {solution-start} missing-mysteries
:class: dropdown
:::

The NaN values look very much like standard Pandas signals of missing (Not Available) data. First, these values arose from loading a data file directly. It's possible that some calculation that led to this data file had invalid floating point values, but it's difficult to see what these might be, or why these could not be avoided. On the other hand, looking at the values, it seems that they occur for smaller countries with less-developed economies (e.g. Aruba) or new countries (at time of data estimate) such as Kosovo. It is easy to see how there might not be good data to calculate e.g. health expenditure per person for these countries.

To explore more, you might have considered looking specifically for rows and columns with many NaN values with something like:

```{code-cell}
missing_hepc = pd.isna(gender_df['health_exp_per_cap'])
gender_df[missing_hepc]
```

(See the [filtering page](0_5_filtering_data_with_pandas) for more on filtering Data Frames.)

For the mean calculation, we notice again that the missing values seem to be
for smaller, less-developed and newer countries. You have already seen that
Pandas assumes that NaN means missing data, and it drops NaN values in
calculations like `.mean()`. Therefore, the result is the mean excluding
these smaller, poorer countries, and the resulting mean will be the mean of
the larger, richer countries. To get a better estimate of worldwide health
expenditure per capita, we might try and think of ways of estimating what the
health expenditure would have been for these smaller countries, perhaps using
other data we do have, from this data frame or elsewhere. At least we should
point out the confounded nature of the `.mean()` as a true estimate.

::: {solution-end}
:::

Pandas supplies us some useful methods for checking missingness.

For instance, we can use `.count()` to show us the number of non-NaN elements in each column:

```{code-cell}
# Count non-NaN (not missing) elements in the Data Frame.
df.count()
```

A useful trick here is to divide the output of the `.count()` method by the
`len()` of the Data Frame. Remember [`len(df)` gives you the number of rows
(number of observations)](len-df). This provides a handy summary of the
_proportion_ of NaNs in each column of the Data Frame:

```{code-cell}
# Show the proportion of missing values, in each column.
# The division operates elementwise, dividing all values above by the divisor.
df.count() / len(df)
```

If we want to use brute force, we can use the `.dropna()` method to remove _any rows_ which have a single NaN value:

```{code-cell}
# Remove the NaN values
df_no_NaN = df.dropna()
df_no_NaN
```

It turns out in this dataset, every row has at least one NaN value so dropping every row with a NaN has dropped _everything_...(we did say this method was brute force)!

+++

**By far the most important thing to know about missing data in Pandas is that
by default NaN values will be _ignored_ in numerical operations**.

Let's look at the `Fertility Rate` column, which contains numerical data:

```{code-cell}
# Show the column
df['Fertility Rate']
```

Because we are dealing with just one column, we can safely use `.dropna()` without losing every row (because not every row of the _Series_ contains a NaN value):

```{code-cell}
# Show the column
df['Fertility Rate'].dropna()
```

Let's compare computing a statistic (the mean) when we drop the NaN values from this Series versus when we leave them in.

+++

Let's first select all the rows relating to (indexed as) Zimbabwe (`'ZWE'`), and the corresponding `'Fertility Rate'` values:

```{code-cell}
# Rows with index value 'ZWE', 'Fertility Rate' column.
zwe_fert = df.loc['ZWE', 'Fertility Rate']
zwe_fert
```

When we use the `.mean()` method on just this column, we get the following value:

```{code-cell}
# Calculate a mean with NaN data included (NaNs will be ignored).
zwe_fert.mean()
```

Using `.dropna()` on this column returns _exactly the same value_ - because by default Pandas will ignore NaNs in numerical operations:

```{code-cell}
zwe_no_nans = zwe_fert.dropna()
zwe_no_nans
```

```{code-cell}
# Drop NaNs gives the same mean
zwe_no_nans.mean()
```

So:

- For NumPy, NaNs propagate, because they indicate an _invalid value_.
- For Pandas, NaNs do not propagate, because they indicate a _missing value_.

Put another way:

- NumPy treats NaNs as _numerical_ indicators of an invalid operation.
- Pandas treats NaNs as _statistical_ indicators of missing data.

This fits with the package names; NumPy for _numerical Python_, Pandas for
_Panel data_ and therefore, statistics.

This difference in NaN handling is a key and important difference between NumPy
and Pandas statistical routines. NumPy `mean`, `min`, `max` and `std` return
NaN, by default, if there are any NaN values in the array.

```{code-cell}
np.std(zwe_fert.values)
```

In contrast, the matching routines in Pandas silently drop the NaN values before calculating `mean`, `min`, `max` and so on.

```{code-cell}
zwe_fert.std()
```

## Summary

On this page we have seen how NaN values indicate different things in NumPy and
Pandas.

In NumPy, NaN values have a _numerical_ meaning, and typically result from
invalid computations, such as dividing zero by zero.

In Pandas, NaN values have _statistical_ meaning. They are most commonly flags for
missing data. By default, these NaN values will be ignored when you call
Pandas' statistical methods for Series or Data Frames.
