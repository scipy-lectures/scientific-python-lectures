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

# Indexing by label and position

## Indexing into Series

From the [What is a Series section](what-is-a-series), remember our maxim:

> A _Series_ is the association of:
>
> - An array of values (`.values`)
> - A sequence of labels for each value (`.index`)
> - A name (which can be `None`).

On this page, we think particularly about the Index (row labels) for Series
and Data Frames. We also discuss the Index that Pandas creates if you do not
specify one.

The default Index that Pandas makes reminds us of the differences between
label indexing (using `.loc`) and position (integer) indexing (using `.iloc`).

Along the way, we'll often press you never to use direct indexing on Series,
as there is still some dangerous ambiguity as to whether you are doing label
or position indexing.

Because it is easy to get mixed up about position (`.iloc`) and label (`.loc`)
indexing, it is often sensible to replace Pandas' default index with a custom
index, to avoid accidental errors when indexing.

## Getting started

```{code-cell}
# import libraries
import numpy as np
import pandas as pd
```

We'll use the [fertility and Human Development Index data once
more](data/data_notes).

```{code-cell}
# Three letter codes for each country
country_codes_array = np.array(['AUS', 'BRA', 'CAN',
                                'CHN', 'DEU', 'ESP',
                                'FRA', 'GBR', 'IND',
                                'ITA', 'JPN', 'KOR',
                                'MEX', 'RUS', 'USA'])
```

```{code-cell}
# Human Development Index Scores for each country
hdis_array = np.array([0.896, 0.668, 0.89,
                       0.586, 0.89,  0.828,
                       0.844, 0.863, 0.49,
                       0.842, 0.883, 0.824,
                       0.709, 0.733, 0.894])
```

## Slicing Series with `.iloc` and `.loc`

```{code-cell}
hdi_series = pd.Series(hdis_array, index=country_codes_array)
hdi_series
```

There is a fundamental difference between the behaviors of `.iloc` and `.loc`
when slicing.

Standard slicing in Python uses integers to specify positions, and gives the
elements _starting at_ the start position, _up to but not including_ the stop
position.

```{code-cell}
my_name = 'Peter Rush'
# From character at position 2, up to (not including) position 7.
my_name[2:7]
```

The same rule applies to indexing Python lists, or NumPy arrays:

```{code-cell}
# From element at position 2, up to (not including) position 7.
country_codes_array[2:7]
```

`.iloc` is indexing by _position_, so it may not be surprising that it slices using the same rules as by-position indexing in NumPy:

```{code-cell}
# From element at position 2, up to (not including) position 7.
hdi_series.iloc[2:7]
```

Now consider slicing by _label_. The _start_ and _stop_ values are no longer
positions, but labels. The label at position 2 is `'CAN'`. The label at
position 7 is the until-recently-European country`'GBR'`.

Here's what we get from slicing using `.loc`:

```{code-cell}
# From element labeled 'CAN', up to (including) element labeled 'GBR'
hdi_series.loc['CAN':'GBR']
```

First notice that label indexing uses values from the Index as start and stop. Unlike NumPy or `.iloc` indexing, which by definition have integers as start and stop (because these are positions), `.loc` indexing start and stop values must match the values in the Index. In this case, the Index has `str` values, so the start and stop values are also `str`.

Second, notice that we got one more value from `.loc` indexing into the
Series, because `.loc` slicing — unlike `.iloc` or NumPy indexing — _includes_
the stop value.

In the last cell, using `.loc`, `'GBR'` was the stop value, and we got the
element corresponding to `'GBR'`.

This is a major difference from NumPy and `.iloc` behavior.

::: {note}

**Stop and `.loc`**

Why does `.loc` slicing return the label corresponding to the stop value, instead of going _up to but not including_ the stop value, like NumPy or `.iloc`?

We should say that this is absolutely the right choice. But why?

Please consider reflecting before reading on.

[Elevator Muzak while you reflect](https://www.youtube.com/watch?v=XlDdrrFY4Ug)

Please click the link above to get you into a reflective mood.

Back to slicing; let's consider the problem of selecting some elements that you
want. You can see the Index. In your case you want all the elements from
`CAN` through `GBR`. When the result includes the stop label, then its obvious
what to do; you do what you do above: `hdi_series.loc['CAN':'GBR']`.

Now consider the alternative — where slicing gives you the elements _up to but
not including_ the stop value. Your problem now becomes annoying and
error-prone. You have to look at the index, identify the last label for the
element you do want (`'GBR'`) and then go one element further, and get the
label for the element _after_ the one you want (in this case `'IND'`. In an
alternative world, where `.loc` was _up to and not including_ the stop value,
indexing to get elements `'CAN'` through `'GBR'` would be
`hdi_series.loc['CAN':'IND']`. Now imagine that for some reason I had deleted
the `'IND'` element, so the following element label is `'ITA'`. In that case,
despite the fact nothing had changed in the elements I'm interested in, I now
have to write `hdi_series.loc['CAN':'ITA']` to get the exact same elements.

So, yes, it's important to remember this difference, but a little reflection
should reveal that this was still the right choice.

:::

+++

## Index labels need not be unique

We haven't specified so far, but there is no general requirement for Pandas
Index values to be unique. Consider the following Series:

```{code-cell}
not_unique_labels = pd.Series(['France', 'Italy', 'UK', 'Great Britain'],
                              index=['FRA', 'ITA', 'GBR', 'GBR'])
not_unique_labels
```

Doing `.loc` indexing with a label that only matches one element gives the
corresponding value:

```{code-cell}
not_unique_labels.loc['FRA']
```

`.loc` matching a label with more than one element returns a subset of the
Series:

```{code-cell}
not_unique_labels.loc['GBR']
```

This can lead to confusing outputs if you don't keep track of whether the Index values uniquely identify the element.

+++

## The default index

Thus far, we have specified the Index in building Series:

```{code-cell}
hdi_series = pd.Series(hdis_array, index=country_codes_array)
hdi_series
```

Pandas allows us to build a Series without specifying an Index:

```{code-cell}
# Make a Series from `hdis_array`, without specifying `index`.
hdi_series_def_index = pd.Series(hdis_array)
hdi_series_def_index
```

Where we did not specify an Index, Pandas has automatically generated one. As
you can see, Pandas displays this default index as a sequence of integers,
starting at 0, and going up to the number of elements minus 1.

+++

Let's take a closer look at the default Index:

```{code-cell}
# The default Pandas index
hdi_series_def_index.index
```

`RangeIndex` is similar to Python's `range`; it is a space-saving container
that represents a sequence of integers from a start value up to, but not
including a stop value, with an optional step size. Here `RangeIndex`
represents the numbers 0 through 14, just as `range` can represent the numbers
0 through 14:

```{code-cell}
zero_through_14 = range(0, 15)
zero_through_14
```

As for `range` we can ask the `RangeIndex` container to give up these numbers
(by iteration) into another container, such as an array or list:

```{code-cell}
# Iterating through `RangeIndex` to give the represented numbers.
np.array(hdi_series_def_index.index)
```

```{code-cell}
# Iterating through a `range` to give the represented numbers.
np.array(zero_through_14)
```

As for `range`, one can ask for the implied elements by indexing:

```{code-cell}
# View the fifth element of the RangeIndex.
fifth_element = hdi_series_def_index.index[4]
fifth_element
```

Notice that the elements from `RangeIndex` are `int`s:

```{code-cell}
type(fifth_element)
```

For all practical purposes, you can treat this `RangeIndex` as being equivalent
to the corresponding sequential NumPy integer array.

::: {exercise-start}
:label: range-index
:class: dropdown
:::

Let's make another Series where we do not specify the index:

```{code-cell}
a_series = pd.Series([1000, 999, 101, 199, 99])
a_series
```

As you have seen, you will have got the default `.index`, a `RangeIndex`:

```{code-cell}
a_series.index
```

What do you expect to see for `list(a_series)`? Reflect, then uncomment below
and try it:

```{code-cell}
# list(a_series)
```

What do you expect to see for `list(a_series.index)`? Reflect, then try it:

```{code-cell}
# list(a_series.index)
```

The Series method `.sort_values` returns a new Series sorted by the values.

```{code-cell}
sorted_series = a_series.sort_values()
```

Now what do you expect to see for `list(sorted_series)`? Reflect, then
uncomment below and try it:

```{code-cell}
# list(sorted_series)
```

How about `list(sorted_series.index)`? Reflect, try:

```{code-cell}
# list(sorted_series.index)
```

What kind of thing do you think the `.index` is now? Reflect and then:

```{code-cell}
# type(sorted_series.index)
```

Can you explain the result of the last cell?

::: {exercise-end}
:::

::: {solution-start} range-index
:class: dropdown
:::

`list` applied to the Series gives a list of the `.values`. List applied to
the `.index` gives a list of the values implied by the Index. For
a `RangeIndex`, this iterates over the Index, extracting all the implied
values.

You should have discovered that the `sorted_series` now has an Index of
integers, and no longer has a `RangeIndex`. The question was prompting you to
reflect that Pandas can only use `RangeIndex` as a space-saving device if the
integers continue to be representable as an ordered sequence with equal steps.
Otherwise it will have to rebuild an array of integers to represent the index.

::: {solution-end}
:::

## Why an Index of integers can be confusing

To recap: for our first few Series, we've used three-letter country codes as
the elements of an `index`. We've just seen what happens if we construct
a Data Frame without telling Pandas what to use as an `index` - it will create
a default `RangeIndex`. `RangeIndex` represents a series of integers.

If you did the exercise, you will have found that Pandas can use `RangeIndex`
when the index is a regular sequence of integers, but must otherwise change to
having an index with an array containing integers, that are the value labels.

What is the advantage of using an index with values that aren't integers
— such as strings? Below are some potential pitfalls to be aware of when using
the default index, and any other index made up of integers.

Let's say we want to access the fifth element of the Series. This is at
integer location 4, because we count from 0. At the moment the numerical
labels implied by the `RangeIndex` "line up" with the integer-based locations:

```{code-cell}
# Show the whole Series
hdi_series_def_index
```

If you somehow ask for element `4`, there is no ambiguity about which element
you mean, because the value with label `4` is also the element at integer
position `4`. Therefore, if we use integer indexing (`.iloc`) we get the same
value as if we use label based indexing (`.loc`):

```{code-cell}
# Indexing using integer location
hdi_series_def_index.iloc[4]
```

```{code-cell}
# Indexing using labels (from the default index)
hdi_series_def_index.loc[4]
```

**Because of this potential for confusion, we strongly suggest that you index
Series with `.loc` and `.iloc`, to be explicit about whether you mean label or
position indexing.**

+++

## Why you should never use direct indexing on Series

[Direct indexing](direct-indirect) occurs where the indexing bracket `[` directly follows the Series value. Conversely, indirect-indexing is indexing where the indexing bracket `[` follows `.loc` or `.iloc`.

Now consider the situation, that we encourage you never to put yourself in,
where you use direct indexing on a Series. You can't specify what type of
indexing you mean with direct indexing. Do you mean label indexing or position
indexing? Pandas will have to make assumptions, and these assumptions may well
be wrong for what you intend. Did we mention, you should never use direct
indexing on Series?

OK, let's imagine that you decided we were being too strict, and used direct
indexing on the Series above, with (implied) integer Index values.

```{code-cell}
# Direct indexing on a Series.  You should never do this.
hdi_series_def_index[4]
```

At the moment, because the positions and integer element labels match up, there
is no ambiguity as to what `4` refers to, so it may not be surprising that
`.iloc`, `.loc` and direct indexing all give the same result.

**But this will not always be the case.** It is extremely common for you to do
operations on the Series — such as sorting and filtering — that will mean that
the integer labels no longer correspond to positions.

For instance let's sort the data in our `hdi_series_def_index` Series in
ascending order. To do this we will use the `.sort_values()` method. We will
cover Pandas methods in detail on [later
pages](0_2_pandas_dataframes_attributes_methods). The `.sort_values()` method
sorts the values of the Series in ascending order, taking the matching labels
in the index with it.

```{code-cell}
# Sorting the *values* in ascending order
hdi_series_def_index_sorted = hdi_series_def_index.sort_values()
hdi_series_def_index_sorted
```

Look at the left hand side of the display from the cell above — in particular,
look at the Index. The numbers within the Index no longer run sequentially
from 0 to 14. This means that the integer position of each element in the
Series no longer matches up with the index label. This can be a potential
source of errors.

::: {note}

**The index type can change if you rearrange elements**

If you haven't done the exercise above, please consider doing it.

If you have, you will have found already that the sorted Series has a new
Index, that is no longer a `RangeIndex` (because the integer labels now cannot
be represented as a regular sequence of integers). Thus
`type(hdi_series_def_index_sorted.index)` will be of type `Index`, rather than
`RangeIndex`.

:::

Let's see what happens if we try to access the fifth element of the series
using integer based indexing (`.iloc[4]`) location based indexing (`.loc[4]`)
and direct indexing (`[4]`) as we did above.

(Did we already say — you should never use direct indexing on Series?)

As you remember, when we did this on the data before sorting, all these
methods returned the same value. Now, however:

```{code-cell}
# Integer indexing on the sorted data
# This is the fifth element in the Series.
hdi_series_def_index_sorted.iloc[4]
```

```{code-cell}
# Label indexing on the sorted data
# This is the element with the label `4`.
hdi_series_def_index_sorted.loc[4]
```

```{code-cell}
# Direct indexing on the sorted data
# Which is this?  Position or label?
# By the way - you should never use direct indexing on Series.
hdi_series_def_index_sorted[4]
```

We have used the number 4 with each indexing method, yet have gotten back
different values for `.iloc` compared to `.loc` and direct indexing.

+++

## Consider specifying a not-default index for Series and Data Frames

We saw above that the default index can induce confusion between label and
position.

If you do avoid using direct indexing, the confusion is less — it will be
easier to remember that `.loc` is for labels and `.iloc` is for positions. But
still, with a little inattention, or some [sloppy
vibe-coding](https://dictionary.cambridge.org/dictionary/english/tautology),
it is nevertheless easy to forget which is which. This is a pitfall of using
sequential numbers as the index — as generated, for example, by `RangeIndex`
— it can lead to confusing results when the position in the sequence and the
`int` label of an element of the Series do not match up.

Compare this to our `hdi_series` which uses the three-letter country codes as
its index:

```{code-cell}
# Show the `hdi_series`.
hdi_series
```

Let's get the fifth element using integer based (`.iloc`) indexing:

```{code-cell}
# Integer (position) indexing
hdi_series.iloc[4]
```

... and let's try to use `.loc[4]` on this Series (this will generate an
error):

```{code-cell}
:tags: [raises-exception]

# Label indexing raises a KeyError ...
hdi_series.loc[4]
```

This `KeyError` tells us that there is no index label `4` (which makes sense
as the index labels in this Series are three-letter country codes). To use
`.loc` with this Series, we must use the three-letter country code strings:

```{code-cell}
# Label based indexing
hdi_series.loc['DEU']
```

It is much harder to get confused when using integer indices as long as you
stick with _indirect indexing_ (`.loc` and `.iloc`). You've specified what you
mean (by label or by position) using the name of the method. However, things
can get dangerously confusing if you use an integer index and _direct
indexing_. Which is why you should not use direct indexing with Series.

Just to remind you, `hdi_series` has the country codes (strings like `'DEU'`)
as the index.

Now, consider, what would happen if we used an integer for _direct indexing_?
As in something like `hdi_series[4]`? Because we haven't
specified that we want to index with labels (`.loc`) or positions (`.iloc`),
Pandas has to make some decision as to how to proceed.

::: {exercise-start}
:label: direct-indexing-series
:class: dropdown
:::

We assume you've just read the text above the exercise, where we consider what
you would expect to happen if:

- Your Series has a index of strings.
- You use direct indexing on this Series with an integer.

As in `hdi_series[4]`. (Don't try it yet).

Pause and reflect what decision you would make in this situation, if you were
a Pandas developer, deciding what Pandas should do. What are the options? Why
would you chose one option over another?

::: {exercise-end}
:::

::: {solution-start} direct-indexing-series
:class: dropdown
:::

Briefly you have two options we could think of as the Pandas developer. You
could:

- Assume that the user is trying to index by label, and raise an error to say
  that the label `4` is not in the index (because your index is a set of
  strings).
- Assume that the user is trying to index by position (Pandas' behavior in
  versions prior to 3).
- Try to persuade the user not to use direct-indexing on Series.

However, there's a big problem with the second option, assuming that the user
is trying to index by position. As you have seen above, in general Pandas
treats direct indexing as by label. So, if there are integer labels, it will,
without complaint, give you the value corresponding the integer label, not the
position. This means that you sometimes treat direct indexing as by label
(when there is an integer index and integer value between the `[]`), and other
times as by position (when there is a non-integer index and integer value
between the `[]`).

In fact, Pandas initially went for this second option, because, if you keep
track of whether your index is an integer index or not, it can be convenient
to avoid the `.iloc` and use direct indexing for position-based indexing (on
a Series with a non-integer index). But recently (and as of Pandas version
3), the Pandas developers [have rethought this
decision](https://github.com/pandas-dev/pandas/issues/49612).

::: {solution-end}
:::

+++

You are about to see the result direct indexing on a Series. In older
versions of Pandas (before version 3) this did something frightening, which
was to _guess_ whether we meant to do `.loc` or `.iloc` indexing depending on
whether the index values are integers. Version 3 takes (in our view) a more
explicit view — and always assumes direct indexing is on labels (`.loc`).

As you have already seen above, if the index consists of integers, and you
specify integers in your direct indexing, then Pandas will assume you mean the
values to be labels (like `.loc`).

If the index does not consist of integers, and you specify integers in your
direct indexing, then the result depends on the version of Pandas you are running. Current versions (version 3 or greater) will raise an error, assuming you meant to index by label (`loc` behavior). Previous versions assumed you meant the values to be positions (like `.iloc`), but would give you a warning about the upcoming change in version 3.

```{code-cell}
:tags: [raises-exception]

# Direct indexing
hdi_series[4]
```

Using a custom non-integer index (e.g. the three-letter country codes) rather
than the default `RangeIndex`, or some other integer index, has the advantage
of avoiding potential confusion (by you, or someone reading the code) between
the integer location of an element, and the index label of that element.

To demonstrate this, let's sort our `hdi_series` in ascending order:

```{code-cell}
# Sorting the Series in ascending order
hdi_series_sorted = hdi_series.sort_values()
hdi_series_sorted
```

The use of custom string-based labels in the index (e.g. `FRA`, `AUS` etc)
avoids confusing misalignment between the default numerical labels and integer
location.

+++

We've said it before, we say it again here — we suggest you _always_ specify
`.loc` or `.iloc` when indexing a Series, in order not to confuse yourself and
your readers as to whether you mean to index by label or position. In this
case `.loc` means we need to use a string, preventing confusion (in Pandas 3)
and errors (Pandas 2) where we use a number and return data we do not expect.

```{code-cell}
# Label-based indexing
hdi_series_sorted.loc['DEU']
```

::: {warning}

**Direct indexing is consistent in older Pandas**

If you're using the latest Pandas (version >= 3), then you should find
indexing is explicit and consistent. You need read no further in this
warning.

However, if you're still using Pandas <3, there are more inconsistencies.

As Pandas was shifting towards more explicit choice of labels over positions
in direct indexing, there were remaining inconsistencies. These were resolved
in version, so if you want to avoid confusion, skip the rest of this note, and
remember _never use direct indexing on a Series_.

If you got this far, we admire your courage. This warning is only to say that
Pandas currently treats _slices_ in direct indexing differently from
individual positions or labels. Specifically, at the moment, it will always
assume integers in slices are positions and not labels. Try some experiments
with `hdi_series[:5]` (string label Series) and `hdi_series_def_index[:5]`
(integer label Series).

See [this Pandas Github
issue](https://github.com/pandas-dev/pandas/issues/49612) for discussion if
you're interested.

If you're using Pandas version 2, you may be confused after trying the
experiments above. Summary for new and old versions — always use `.iloc` and
`.loc` to avoid ambiguity.

:::

(loc-iloc-df)=

## `.loc` and `.iloc` with Data Frames

So far we have spent much time with `.loc` and `.iloc` on Series, but less
time on `.loc` and `.iloc` for Data Frames.

Series are like one-dimensional arrays (with and Index and a Name) - therefore
`.loc` and `.iloc` indexing into Series looks like indexing into
one-dimensional NumPy arrays.

A Data Frame is like a two dimensional array, so `.loc` and `.iloc` indexing
looks like indexing into two-dimensional NumPy arrays.

Consider the following two-dimensional NumPy array:

```{code-cell}
hdi_series[:5]
```

```{code-cell}
hdi_series_def_index[:5]
```

```{code-cell}
two_d_arr = np.array([[1, 2, 3], [11, 21, 31], [101, 102, 103]])
two_d_arr
```

If we index with one expression between the indexing brackets, we select
_rows_:

```{code-cell}
# Select the second row.
two_d_arr[1]
```

If we want to select columns, we must specify two indexing expressions between
the indexing brackets, separated by a comma:

```{code-cell}
# Select the second row, third column.
two_d_arr[1, 2]
```

As usual, we can use slices as indexing expressions (e.g. expressions
containing colons `:`):

```{code-cell}
# Select first and second rows, second and third columns.
two_d_arr[:2, 1:3]
```

```{code-cell}
# Select all rows, third column.
two_d_arr[:, 2]
```

Because a Data Frame has rows and columns, it corresponds to a two-dimensional
array.

Let us make an example Data Frame for illustration. In fact we'll return to the Data Frame from [the introduction to `pd.DataFrame`](pd-data-frame-intro).

```{code-cell}
# Fertility rate scores for each country
fert_rates_array = np.array([1.764, 2.247, 1.51,
                             1.628, 1.386, 1.21,
                             1.876, 1.641, 3.35,
                             1.249, 1.346, 1.467,
                             2.714, 1.19 , 2.03 ])
# Series from array.
fert_rate_series = pd.Series(fert_rates_array, index=country_codes_array)

# Data Frame from dict of Series.
example_df = pd.DataFrame({'Human Development Index': hdi_series,
                           'Fertility Rate': fert_rate_series})
example_df
```

If we ask for the Data Frame `.values`, we get a two-dimensional NumPy array:

```{code-cell}
example_df.values
```

When direct indexing with `.loc` or `.iloc`, we can select rows with a single
indexing expression:

```{code-cell}
# Select row corresponding to label 'RUS'
example_df.loc['RUS']
```

```{code-cell}
# Select rows from that labeled 'ITA' to that labeled 'RUS'.
# Remember, `.loc` is inclusive of the stop value.
example_df.loc['ITA':'RUS']
```

```{code-cell}
# Select second row by position.
example_df.iloc[1]
```

```{code-cell}
# Select second through fifth row by position.
# As standard for Python integers indexing, this is exclusive of stop position.
example_df.iloc[1:5]
```

Like the NumPy two-dimension indexing case, if we want to select columns with
`.loc` or `.iloc`, we must give two indexing expressions, separated by
a comma:

```{code-cell}
# Select rows 'ITA' through 'RUS', 'Fertility Rate' column.
example_df.loc['ITA':'RUS', 'Fertility Rate']
```

```{code-cell}
# Row for 'RUS', all columns.
example_df.loc['RUS', :]
```

```{code-cell}
# Select second through fifth row by position, first column by position.
example_df.iloc[1:5, 0]
```

```{code-cell}
# Second row, all columns.
example_df.iloc[1, :]
```

## The catechism of Pandas indexing

We are now ready for the definitive advice for your life using indexing in Pandas.

1. Never use direct indexing on Series. Always use indirect indexing (`.loc`
   and `.iloc`).
1. You _can and should_ use direct indexing on Data Frames, but in two and
   only two specific cases. These are:
   1. _Direct indexing with a column name_, or sequence of column names. Here
      the column name (label) or sequence of column names follows the Data
      Frame value and the opening `[` — as in:

      ```python
      example_df['Human Development Index']
      ```

      and

      ```python
      example_df[['Human Development Index', 'Fertility Rate']
      ```

   1. _Direct indexing with a Boolean Series_. See [the filtering
      page](0_5_filtering_data_with_pandas) for much more on Boolean Series
      and indexing. The Boolean Series follows the data frame value and the
      opening `[`, and selects rows for which the Boolean Series has True
      values — as in:

      ```python
      # Make a Boolean Series.
      have_high_hdi = example_df['Human Development Index'] > 0.6
      # Select rows by indexing with Boolean Series.
      high_df = example_df[have_high_hdi]
      ```

We strongly suggest that you restrict your use of direct indexing to a) Data
Frames (not Series) and b) these specific cases. We do the same.

+++

## Summary

On this page we have looked at the Pandas Index, and different ways of
indexing into Pandas Series.

We discussed the default index that Pandas provides, of integer labels, and we
showed how to get Series values by label (`.loc`) and by position (`.iloc`).

`.loc` differs from `.iloc` and other Python indexing in that slices _include
their stop value_.

We pressed you to completely avoid using direct indexing on Pandas Series, because of the potent confusion that can arise between label and position indexing.

For best results, you should specify an interpretable `index` for your Series
and Data Frames.

Direct indexing into Data Frames is common and useful, in two and only two
situations:

1. Direct indexing using a column name or sequence of names.
1. Direct indexing using a Boolean Series (see [filtering
   page](0_5_filtering_data_with_pandas)).

We can use `.loc` and `.iloc` on Data Frames, remembering that this indexing
acts like indexing two-dimensional NumPy arrays; when selecting columns, we
first need to specify a selection for rows.
