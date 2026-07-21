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

# Indexing by label, indexing by position

## Indexing and Series

From the [What is a Series section](what-is-a-series), remember our maxim:

> A *Series* is the association of:
>
> * An array of values (`.values`)
> * A sequence of labels for each value (`.index`)
> * A name (which can be `None`).

On this page we take a brief look at Series `.name`s, and show that they can be useful when created Data Frames from Series.

Then we look at the Index for Series and Data Frames, and the default index
that Pandas creates if you do not specify one.

The default Index that Pandas makes reminds us of the differences between
label indexing (using `.loc`) and position (integer) indexing (using `.iloc`);
we show that current Pandas allows some ambiguity about what type of indexing
you are doing when using *direct indexing*.  It is sensible to replace Pandas'
default index with a custom index, to avoid accidental errors when indexing.

## Getting started

```{code-cell} ipython3
# import libraries
import numpy as np
import pandas as pd
```

We'll use the [fertility and Human Development Index data once
more](data/data_notes).

```{code-cell} ipython3
# Three letter codes for each country
country_codes_array = np.array(['AUS', 'BRA', 'CAN',
                                'CHN', 'DEU', 'ESP',
                                'FRA', 'GBR', 'IND',
                                'ITA', 'JPN', 'KOR',
                                'MEX', 'RUS', 'USA'])
```

```{code-cell} ipython3
# Human Development Index Scores for each country
hdis_array = np.array([0.896, 0.668, 0.89,
                       0.586, 0.89,  0.828,
                       0.844, 0.863, 0.49,
                       0.842, 0.883, 0.824,
                       0.709, 0.733, 0.894])
```

## What's in a `name`?

You may remember from the [](series-names) section that Series have optional names, where the default name is `None`.

To recap, here we build a new Series by passing the values as the first argument, and the labels (Index) as the second:

```{code-cell} ipython3
# Make a series from the `hdis_array` array.
hdi_series = pd.Series(hdis_array, country_codes_array)
hdi_series
```

In fact we generally prefer to pass the Index as a named argument, to be explicit:

```{code-cell} ipython3
# The same constructor call, with index as a named argument.
hdi_series = pd.Series(hdis_array, index=country_codes_array)
```

Currently, our Series has data values (`values`) and an `index`, but the
`name` attribute is None (the default).

```{code-cell} ipython3
# Show the `values`
hdi_series.values
```

```{code-cell} ipython3
# Show the `index`
hdi_series.index
```

```{code-cell} ipython3
# Show that `name` is None
hdi_series.name is None
```

However, the `name` attribute can be useful for remind you of the nature of
the data in the `values` array.  As you will see below, it's particularly
useful to attach names to Series when you will use the Series to build a Data
Frame.

Let's make a new Series - called `hdi_series_with_name_and_index` - where we
**do** specify a `name` attribute when calling the `pd.Series()` constructor.

```{code-cell} ipython3
# Make a series from the `hdis` array, specifying the `name` attribute.
hdi_series_with_name_and_index = pd.Series(hdis_array,
                                           index=country_codes_array,
                                           name='Human Development Index')
hdi_series_with_name_and_index
```

```{code-cell} ipython3
# Show the `name` attribute
hdi_series_with_name_and_index.name
```

## What's in an `index`?

We have seen what happens if we make a Series with no `name` attribute. But
what if we make a Series without specifying an `index`?

```{code-cell} ipython3
# Make a Series from `hdis_array`, without specifying `index` or `name`.
hdi_series_no_name_no_index =  pd.Series(hdis_array)
hdi_series_no_name_no_index
```

Here, you can see that, where we did not specify an `index`, Pandas has
automatically generated one.

+++

Let's take a closer look at this `index` by directly accessing the `.index`
attribute:

```{code-cell} ipython3
# The default Pandas index
hdi_series_no_name_no_index.index
```

`RangeIndex` is very similar to Python's `range`; it is a space-saving
container that represents a sequence of integers from a start value up to, but not including a stop value, with an optional step size.  Here `RangeIndex` represents the numbers 0 through 14, just as `range` can represent the numbers 0 through 14:

```{code-cell} ipython3
zero_through_14 = range(0, 15)
zero_through_14
```

As for `range` we can ask the `RangeIndex` container to give up these numbers (by iteration) into another container, such as an array or list:

```{code-cell} ipython3
# Iterating through `RangeIndex` to give the represented numbers.
np.array(hdi_series_no_name_no_index.index)
```

```{code-cell} ipython3
# Iterating through a `range` to give the represented numbers.
np.array(zero_through_14)
```

As for `range`, one can ask for the implied elements by indexing:

```{code-cell} ipython3
# View the fifth element of the RangeIndex
fifth_element = hdi_series_no_name_no_index.index[4]
fifth_element
```

Notice that the elements from `RangeIndex` are `int`s:

```{code-cell} ipython3
type(fifth_element)
```

For all practical purposes, you can treat this `RangeIndex` as being
equivalent the corresponding sequential Numpy integer array.

::: {exercise-start}
:label: range-index
:class: dropdown
:::

Let's make another Series where we do not specify the index:

```{code-cell} ipython3
a_series = pd.Series([1000, 999, 101, 199, 99])
a_series
```

As you have seen, you will have got the default `.index`, a `RangeIndex`:

```{code-cell} ipython3
a_series.index
```

What do you expect to see for `list(a_series)`?  Reflect, then uncomment below
and try it:

```{code-cell} ipython3
# list(a_series)
```

What do you expect to see for `list(a_series.index)`?  Reflect, then try it:

```{code-cell} ipython3
# list(a_series.index)
```

The Series method `.sort_values` returns a new Series sorted by the values.

```{code-cell} ipython3
sorted_series = a_series.sort_values()
```

Now what do you expect to see for `list(sorted_series)`?  Reflect, then
uncomment below and try it:

```{code-cell} ipython3
# list(sorted_series)
```

How about `list(sorted_series.index)`?  Reflect, try:

```{code-cell} ipython3
# list(sorted_series.index)
```

What kind of thing do you think the `.index` is now?  Reflect and then:

```{code-cell} ipython3
# type(sorted_series.index)
```

Can you explain the result of the last cell?

::: {exercise-end}
:::

::: {solution-start} range-index
:class: dropdown
:::

You should have discovered that the `sorted_series` now has an Index of
integers, and no longer has a `RangeIndex`.  The question was prompting you to
reflect that Pandas can only use `RangeIndex` as a space-saving device if the
integers continue to be representable as an ordered sequence with equal steps.
Otherwise it will have to rebuild an array of integers to represent the index.

::: {solution-end}
:::

## Why an index of integers can be confusing

To recap: we've used three-letter country codes as the elements of an `index`,
and we've just seen what happens if we construct a Data Frame without telling
Pandas what to use as an `index` - it will create a default `RangeIndex`.  `RangeIndex` represents a series of integers.

If you did the exercise, you will have found that Pandas can us `RangeIndex`
when the index is a regular sequence of integers, but must otherwise change to
having an index with an array containing integers, that are the value labels.

What is the advantage of using an index with values that aren't integers
— such as strings? Below are some potential pitfalls to be aware of when using
the default index, and any other index made up of integers.

Let's say we want to access the fifth element of the Series. This is at
integer location 4, because we count from 0. At the moment the numerical
labels implied by the `RangeIndex` "line up" with the integer-based locations:

```{code-cell} ipython3
# Show the whole Series
hdi_series_no_name_no_index
```

If you ask for element `4`, there is no ambiguity about which element you
mean, because the value with label `4` is also the element at integer position
`4`.  Therefore, if we use integer indexing (`.iloc`) we get the same value as
if we use label based indexing (`.loc`):

```{code-cell} ipython3
# Indexing using integer location
hdi_series_no_name_no_index.iloc[4]
```

```{code-cell} ipython3
# Indexing using labels (from the default index)
hdi_series_no_name_no_index.loc[4]
```

What if we don't tell Pandas what type of indexing we want to do?  Meaning, we
do not use `.iloc` or `.loc`, we just use the sort of [direct
indexing](direct-indirect) we would use on a Python list or array?

```{code-cell} ipython3
# Direct indexing
hdi_series_no_name_no_index[4]
```

There is no ambiguity as to what `4` refers to, so it may not be surprising that `.iloc`, `.loc` and direct indexing all give the same result.

**But this will not always be the case.** Certain functions and methods that
we may want to use to sort and organise our data will cause cause misalignment
between the *integer labels* and the *integer position* of a given element of
the Series.

For instance let's sort the data in our `hdi_series_no_name_no_index` Series
in ascending order.  To do this we will use the `.sort_values()` method. We
will cover Pandas methods in detail on [later
pages](0_2_pandas_dataframes_attributes_methods.md). For now the
`.sort_values()` method sorts the values of the Series in ascending order, taking the matching labels in the index with it.

```{code-cell} ipython3
# Sorting the *values* in ascending order
hdi_series_no_name_no_index_sorted = hdi_series_no_name_no_index.sort_values()
hdi_series_no_name_no_index_sorted
```

Look at the left hand side of the display from the cell above — in particular,
look at the Index.  The numbers within the Index no longer run sequentially
from 0 to 14. This means that the integer location of each element in the
Series no longer matches up with the index label. This can be a potential
source of errors.

::: note

**The index type can change if you rearrange elements**

If you haven't done the exercise above, please consider doing it.

If you have, you will have found already that the sorted Series has a new
Index, that is no longer a `RangeIndex` (because the integer labels now cannot
be represented as a regular sequence of integers):

```{code-cell} ipython3
type(hdi_series_no_name_no_index_sorted.index)
```

:::

Let's see what happens if we try to access the fifth element of the series
using integer based indexing (`.iloc[4]`) location based indexing (`.loc[4]`)
and direct indexing (`[4]`) as we did above.

As you remember, when we did this on the data before sorting, all these
methods returned the same value.  Now, however:

```{code-cell} ipython3
# Integer indexing on the sorted data
# This is the fifth element in the Series.
hdi_series_no_name_no_index_sorted.iloc[4]
```

```{code-cell} ipython3
# Label indexing on the sorted data
# This is the element with the label `4`.
hdi_series_no_name_no_index_sorted.loc[4]
```

```{code-cell} ipython3
# Direct indexing on the sorted data
# Which is this?  Position or label?
hdi_series_no_name_no_index_sorted[4]
```

We have used the number 4 with each indexing method, yet have gotten back
different values for `.iloc` compared to `.loc` and direct indexing.

This is a pitfall of using sequential numbers as the index — as generated, for
example, by `RangeIndex` — it can lead to confusing results when the position
in the sequence and the `int` label of an element of the Series do not match
up.

Compare this to our `hdi_series_with_name_and_index` which uses the
three-letter country codes as its index:

```{code-cell} ipython3
# Show the `hdi_series_with_name_and_index` Series
hdi_series_with_name_and_index
```

Let's get the fifth element using integer based (`.iloc`) indexing:

```{code-cell} ipython3
# Integer indexing
hdi_series_with_name_and_index.iloc[4]
```

...and let's try to use `.loc[4]` on this Series (this will generate an error):

```{code-cell} ipython3
:tags: [raises-exception]

# Label indexing raises a KeyError ...
hdi_series_with_name_and_index.loc[4]
```

This `KeyError` tells us that there is no index label `4` (which makes sense
as the index labels in this Series are three-letter country codes). To use
`.loc` with this Series, we must use the three-letter country code strings:

```{code-cell} ipython3
# Label based indexing
hdi_series_with_name_and_index.loc['DEU']
```

We don't think it's very confusing when using integer indices with `.loc` and
`.iloc`.  You've specified what you mean (by label or by position) with the
name of the method.  However, things can get dangerously confusing if you use
an integer index and *direct indexing*.

Just to remind you, `hdi_series_with_name_and_index` has the country codes (strings like `'DEU'`) as the index.

Now, consider, what would happen if we used an integer for *direct indexing*?
As in something like `hdi_series_with_name_and_index[4]`?  Because we haven't
specified that we want to index with labels (`.loc`) or positions (`.iloc`),
Pandas has to make some decision as to how to proceed.

::: {exercise-start}
:label: direct-indexing-series
:class: dropdown
:::

We assume you've just read the text above the exercise, where we consider what you would expect to happen if:

* Your Series has a index of strings.
* You use direct indexing on this Series with an integer.

As in `hdi_series_with_name_and_index[4]`. (Don't try it yet).

Pause and reflect what decision you would make in this situation, if you were
a Pandas developer, deciding what Pandas should do?  What are the options?
Why would you chose one option over another?

::: {exercise-end}
:::

::: {solution-start} direct-indexing-series
:class: dropdown
:::

Briefly you have two options we could think of as the Pandas developer.  You
could:

* Assume that the user is trying to index by label, and raise an error to say
  that the label `4` is not in the index (because your index is a set of
  strings).
* Assume that the user is trying to index by position (Pandas' current
  behavior).

However, there's a big problem about assuming that the user is trying to index
by position, which is that, as you have seen above, in general Pandas treats
direct indexing as by label.  So, if there are integer labels, it will,
without complaint, give you the value corresponding the integer label, not the
position.  This means that you sometimes treat direct indexing as by label
(when there is an integer index and integer value between the `[]`), and other
times as by position (when there is a non-integer index and integer value
between the `[]`).

As you can see, Pandas initially went for this option, because, if you keep
track of whether your index is an integer index or not, it can be convenient
to avoid the `.iloc` and just do position-based indexing (on a Series with
a non-integer index) with direct indexing.  But recently, the Pandas developers have reconsidered, and are planning to remove this ambiguous use of integer indexing, on the basis that it is too easy to be confused as to which indexing you are doing.

::: {solution-end}
:::

+++

You are about to see that direct indexing on a Series, for now, does something
truly awful, which is to *guess* whether we mean to `.loc` or `.iloc` indexing
depending on whether the index values are integers.

So, as you have already seen above, if the index consists of integers, and you
specify integers in your direct indexing, then Pandas will assume you mean the
values to be labels (like `.loc`).

If the index consists does not consist of integers, and you specify integers
in your direct indexing, then Pandas will assume you mean the values to be
positions (like `.iloc`).

```{code-cell} ipython3
# Direct indexing
hdi_series_with_name_and_index[4]
```

Using a custom non-integer index (e.g. the three-letter country codes) rather
than the default `RangeIndex`, or some other integer index, has the advantage
of avoiding potential confusion between the integer location of an element,
and the index label of that element.

To demonstrate this, let's sort our `hdi_series_with_name_and_index` in ascending order:

```{code-cell} ipython3
# Sorting the Series in ascending order
# Notice the standard Python / Pandas trick of using parentheses to
# break the statement over multiple lines.  This is a common trick, but
# it's particularly common for Pandas processing, to chain multiple
# method calls on the output of previous calls.  You'll see what we mean soon.
hdi_series_with_name_and_index_sorted = (hdi_series_with_name_and_index
                                         .sort_values())
hdi_series_with_name_and_index_sorted
```

The use of custom string-based labels in the index (e.g. `FRA`, `AUS` etc)
avoids confusing misalignment between the default numerical labels and integer
location.

+++

It's good and safe practice to explicitly specify `.loc` or `.iloc` when
indexing a Series, in order not to confuse Pandas as to whether you mean to
index by label or position.   In this case `.loc` means we have to use
a string, preventing errors where we use a number and return data we do not
expect.

```{code-cell} ipython3
# Label-based indexing
hdi_series_with_name_and_index_sorted.loc['DEU']
```

Therefore, a good additional maxim is: **use `.loc` and `iloc` unless there is
a good reason not to!**.

+++

## Strange and `name`less Data Frame columns

Remember, a Data Frame is a dictionary-like collection of Series.

We often create Data Frames with an actual dictionary of Series, or with single Series.

When we build Data Frames from Series, it becomes eminently sensible to
specify a `name` attribute for each Series.

Pandas will not *force* us to do this, but it leads to some error-prone
consequences if we do not. In fact, the default `RangeIndex` crops up again
here, and can create confusion in similar ways to the ones we have seen in the
last section.

As shown above, our `hdi_series` does not have a `name` attribute. Let's pass
this Series to the `pd.DataFrame()` constructor, to see the consequence of
a `name`less Series in this context. We will call the resulting Data Frame
`no_name_df`:

```{code-cell} ipython3
# Verify again that the `hdi_series` has no `name` attribute
hdi_series.name is None
```

```{code-cell} ipython3
# A Data Frame made of a Series with no `name` attribute
no_name_df = pd.DataFrame(hdi_series)
no_name_df
```

Ok, so in the absence of a `name` attribute Pandas has labelled the column with a `0`. If we inspect more deeply, we find that actually Pandas, in the absence of being instructed otherwise, has created a `RangeIndex`, but this time for the `columns` (e.g. the column names) of the Data Frame. If you look at the Data Frame above, you can see that the `index` attribute is the three-letter country codes. However, the `.columns` are a `RangeIndex`:

```{code-cell} ipython3
# Look at the column names via the `.columns` attribute
no_name_df.columns
```

Sure enough, if we check the `type` of the fist element in this `RangeIndex`
it is an `int` — we can think of it not as a column *name* but a number
standing in for a column name:

```{code-cell} ipython3
# Check the type of the first element in the `.columns` attribute
type(no_name_df.columns[0])
```

It may be obvious why this naming convention for Data Frame columns can lead
to errors for reasons of low interpretability. We typically want our column
names to be descriptive of the data in the column, to ourselves and to other
people reading our code. If we do not specify a `name` attribute for our
Series when creating Data Frames, the default numerical column names supplied
by Pandas are hard to interpret, and it is easy to misinterpret, or to forget
what data is in that column, leading to human errors.

They can also lead to indexing errors, similar to those we saw in the previous
section. To demonstrate this, let's compare the `name`less Data Frame above to
a Data Frame created from a Series with a `name`.

Our aptly named `hdi_series_with_name_and_index` has a `name` attribute
— (look at the last line of the output from the cell below, the `Name: Human
Development Index`):

```{code-cell} ipython3
# A Series with a `name`
hdi_series_with_name_and_index
```

Let's call the `pd.DataFrame()` constructor on this Series - we'll call the
resultant Data Frame `named_df`:

```{code-cell} ipython3
# Create a new Data Frame, with a sensible name for the column
named_df = pd.DataFrame(hdi_series_with_name_and_index)
named_df
```

We see that Pandas has automatically used the `name` attribute as the column
name, hugely increasing the interpretability of the resulting Data Frame.

We can see the `name` in the `.columns` attribute of the new  `named_df` Data
Frame:

```{code-cell} ipython3
# Show the column names from the `named_df` Data Frame
named_df.columns
```

Now, let's try using direct indexing with each Data Frame, using the column names.

This is very straightforward for the `named_df`:

```{code-cell} ipython3
# Direct indexing to retrieve a column by name
named_df['Human Development Index']
```

What about for `no_name_df`? Well, the column name there is `0`, of `int`
type.

What happens if use direct indexing? We are hoping we see something like the
output of the cell above, albeit with a `0` for the `name`, rather than `Human
Development Index`:

```{code-cell} ipython3
# Direct indexing with no `name` attribute
no_name_df[0]
```

Sure enough that is what we see. But the operation we have used looks very
much like *integer* indexing on a Series, which will return a single value:

```{code-cell} ipython3
# Integer indexing a Series (without using `.iloc`)
hdi_series[0]
```

This can be confusing.

The numerical label can also be confusing if we introduce more columns, especially if we mix in columns which do have `name` attributes that we specify.

For instance, if we add in the full name of each country:

```{code-cell} ipython3
# Country names array
country_names_array = np.array(['Australia', 'Brazil', 'Canada',
                                'China', 'Germany', 'Spain',
                                'France', 'United Kingdom', 'India',
                                'Italy', 'Japan', 'South Korea',
                                'Mexico', 'Russia', 'United States'])
country_names_array
```

Let's add this into the `no_name_df`, using the `name` `'Country Names'`:

```{code-cell} ipython3
no_name_df['Country Names'] = country_names_array
no_name_df
```

We now have one column that has an `int` as its column name, and another with
a string.

In the same way as for a numerical `index`, this situation can become
confusing if the numerical labels become misaligned with their integer
location in the `.columns` attribute.

For instance, we could reverse the order of the columns, by using *direct indexing* in the Data Frame to request the columns in reverse order by column label:

```{code-cell} ipython3
# Re-arrange the columns
# Specify the order of columns we want.
cols = ['Country Names', 0]
# Use direct indexing to select columns in given order.
reversed_col_df = no_name_df[cols]
reversed_col_df
```

We now have a column with the `name` `0` (an int) that is not at the 0-th location of the `.columns` attribute:

```{code-cell} ipython3
# Show the 0-th element of the `.columns` attribute
reversed_col_df.columns[0]
```

We are now in the precarious situation of using a `int` `0` both as a *label* and as a *location*. In the cell above, it is a location, it the cell below it serves as a label:

```{code-cell} ipython3
# Direct index the `0` column
reversed_col_df[0]
```

Ideally, we would like a clear separation between integer indexes (like `0`)
and *column names*.

This confusing situation can be completely avoided in the case of `named_df`. Let's add the country names to this Data Frame, that thus far only has a single column, containing the "Human Development Index" values.

```{code-cell} ipython3
# Show the Data Frame thus far
named_df
```

```{code-cell} ipython3
# Add the country names
named_df['Country Names'] = country_names_array
named_df
```

We don't introduce any confusion or add propensity to error by re-arranging
these columns which have sensible string names:

```{code-cell} ipython3
# Re-arrange the columns
reversed_named_df = named_df[['Country Names', 'Human Development Index']]
reversed_named_df
```

In fact, now if we want to use integer indexing, we are forced to use `.iloc`,
as we will otherwise get an error:

```{code-cell} ipython3
:tags: [raises-exception]

# This will not work if each column has a `name` string
reversed_named_df[0]
```

This compels us to stick to the good practice maxim we introduced above: **use
`.loc` and `iloc` unless there is a good reason not to!**.

Giving every column a sensible string as a `name` is one situation where direct indexing (e.g. in the present context using `named_df['Human Development Index']` etc.) is safe and non-error prone. As we have seen, issues can arise if we do not specify `name` attributes, and let Pandas automatically generate numeric labels for our Data Frame columns...

+++

## Summary

On this page we have looked at the differences between attributes of the
Pandas Series - the `name` is optional, but a default `RangeIndex` will be
supplied if no custom `index` is specified.

We have seen that the `RangeIndex` and other indices with integer labels can
lead to errors if the numeric row labels become misaligned with the integer
location of a given row. Similarly, if we do not specify a `name` attribute
for Data Frame columns, Pandas will generate numeric labels which can create
confusion between integer-based and direct indexing.

For best results, we should specify both an interpretable `index` and
interpretable `name` attributes for our Series, especially when they are part
of Data Frames.
