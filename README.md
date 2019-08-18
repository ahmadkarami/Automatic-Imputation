# Automatic-Imputation

This module is for imputation missing value in data frame.

this module consideres two type of column: 1. numerical, 2. categorical.
this module uses Regression for numerical columns and classification for categorical columns.

#### very important point #1: categorical column of dara frame must be string not number.
#### very important point #2: All columns must either be numeric or string. If there is a column that contains a combination of numbers and strings, you must either delete that column or convert it to a numeric-only or string-only column.

usage: Impute(dataFrame, countOfNullPerRowToFill)

1. this module takes a dataFrame include only NA for missing value not else.
2. this module is not optimize, therefor small data set should be used.
3. this module impute missing value at categorical column in a row that has only one NA value among categorical columns at the same row.
4. this module impute missing value at numerical column in a row that has up to "countOfNullPerRowToFill" NA value among numerical
   columns at the same row. it means that if user pass 3 to countOfNullPerRowToFill, rows that have 1 NA value among numeric columns,
   2 NA value among numeric columns and 3 NA value among numeric columns will be imputed.
5. this module returns data frame with no NA values
