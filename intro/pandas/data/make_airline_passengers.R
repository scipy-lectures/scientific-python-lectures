# Write `airline_passengers.csv` from R dataset.
#
# Make sure you have run the line below to install the `zoo` package.
# install.packages('zoo')
#
# AirPassengers is a dataset available in base R.
dates <- zoo::as.Date(AirPassengers)
date_col <- strftime(dates, '%Y-%m')
passengers <- as.numeric(AirPassengers)
df <- data.frame(date_col, passengers)
names(df) <- c("Month", "Thousands of Passengers")
write.csv(df, 'airline_passengers.csv', row.names=FALSE)
