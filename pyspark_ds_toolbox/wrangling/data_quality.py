"""Data Quality toolbox.

Module dedicated to provide data quality tools.
"""

from typeguard import typechecked
import pyspark 
from pyspark.sql import functions as F
import pandas as pd


@typechecked
def count_percent_missing_rows_per_column(sdf: pyspark.sql.dataframe.DataFrame) -> pd.core.frame.DataFrame:
  """Computes the percentage of missing values for each column.

  Args:
      dataframe_spark (pyspark.sql.dataframe.DataFrame): A Spark DataFrame.

  Returns:
      pd.core.frame.DataFrame: A Pandas DF where each row representes the sdf columns and the values represents the percentage of null values.
  """
  dataframe_spark_n_rows = sdf.count()
  df_missing_column = sdf\
  .select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in sdf.dtypes if c_type not in ('timestamp', 'date')])\
  .toPandas()
  df_missing_column = round(100*(df_missing_column.rename(index={0: 'percent_missing'}).T.sort_values("percent_missing",ascending=False)/dataframe_spark_n_rows),2)  
  return df_missing_column
