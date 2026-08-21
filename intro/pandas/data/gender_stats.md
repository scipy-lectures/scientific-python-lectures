---
orphan: true
---

# Gender equality statistics

The data are a version of this dataset from the World Bank on gender and
inequality usually housed at [World Bank
gender-statistics](https://data.worldbank.org/data-catalog/gender-statistics).

The version I've used here was from 2017. The 2017 version had data on health
care expenditure that the current data does not have.

You can get a copy of the 2017 data at:
[Gender_StatsData.csv](https://ndownloader.figshare.com/files/17803202).

See [the Gender Statistics dataset
page](https://github.com/matthew-brett/datasets/tree/1ac6d8c/gender_stats) for
more detail on my processing of the data.

In summary, I've rearranged the data, calculated averaged data from 2012 to
2016 for a set of columns I was interested in, and only kept rows corresponding
to actual countries, rather than summary groups like "Arab World" and "Lower
middle income".

The data file is {download}`gender_stats.csv <gender_stats.csv>`.

I have also made a version with fewer columns for the starting pages on Pandas,
called `gender_stats_min.csv`:

```python
import pandas as pd
df = pd.read_csv('gender_stats.csv')
df = df.loc[:, ['country_name', 'country_code', 'gdp_us_billion',
                'mat_mort_ratio', 'population']]
df.to_csv('gender_stats_min.csv', index=None)
```

See [gender stats data dictionary](gender_stats_data_dict) for a list of the
column names and their meaning.

The data is licensed under
[CC-BY](https://creativecommons.org/licenses/by/4.0/). Please attribute to the
World Bank Data Catalogue with the this URL:
<https://datacatalog.worldbank.org/dataset/gender-statistics>.
