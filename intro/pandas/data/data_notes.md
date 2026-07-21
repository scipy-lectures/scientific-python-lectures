---
orphan: true
---

# Datasets and licenses

## `year_2000_hdi_fert.csv`

This derives from the {ref}`children-per-woman-data` with row
selection from data in the [Gender Stats data](gender_stats).  See the
[make_y2k_hdi_fert](make_y2k_hdi_fert) notebook for the exact derivation.

(children-per-woman-data)=
## `children-per-woman-vs-human-development-index` files

See [Children HDI
README](readme_children-per-woman-vs-human-development-index) for source and
citation.

## `gender_stats` files

See [Gender Stats README](gender_stats) for source and citation.

(airline-passengers-data)=
## `airline_passengers.csv`

This is classic dataset from Box and Jenkins' "Time Series Analysis,
Forecasting and Control" book {cite:p}`box2016forecasting`, but first
published in {cite:p}`brown1963smoothing` (p. 429), with the source given as
"FAA Statistical Handbook of Civil Aviation (several annual issues)". The
dataset is available via base R as the [AirPassengers
dataset](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html).

The `airline_passengers.csv` file here derives directly from the R dataset, by
running the {download}`make_airline_passengers.R` script.

## References

```{bibliography}
:filter: docname in docnames
```
