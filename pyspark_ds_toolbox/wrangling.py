"""Data Wrangling toolbox.

Module dedicated to provide data wrangling function.
"""

from typeguard import typechecked
import pyspark 
from pyspark.sql import functions as F
import pandas as pd


@typechecked
def pivot_long(
    dfs: pyspark.sql.dataframe.DataFrame,
    key_column_name: str,
    value_column_name: str,
    key_columns: list,
    value_columns: list,
    print_stack_expr: bool = False
) -> pyspark.sql.dataframe.DataFrame:
    """Function to pivot columns into rows (similar to pandas pivot). See the following link:
    https://sparkbyexamples.com/pyspark/pyspark-pivot-and-unpivot-dataframe/#:~:text=PySpark%20pivot()%20function%20is,individual%20columns%20with%20distinct%20data.

    Args:
        dfs (pyspark.sql.dataframe.DataFrame): A sparkDF.
        key_column_name (str): Name for the column that will receive the columns passed in value_columns as values.
        value_column_name (str): Name the column that will contain the values from the column passed in value_columns.
        key_columns (list): List of columns that will not be 'pivoted'.
        value_columns (list): List os columns to be 'pivoted'
        print_stack_expr (bool, optional): Print the spark sql expression that executes the pivot. Defaults to False.

    Returns:
        [pyspark.sql.dataframe.DataFrame]: The dfs argument with the value_columns 'pivoted'.
    """
    # 1) Creating the stack expression
    len_value_columns = len(value_columns)
    stack_expr = ""
    stack_expr = stack_expr + f'stack({len_value_columns}, '
    for i in range(len_value_columns):
        if i == max(range(len(value_columns))):
            stack_expr = stack_expr + f"'{value_columns[i]}', {value_columns[i]}) as ({key_column_name}, {value_column_name})"
        else:
            stack_expr = stack_expr + f"'{value_columns[i]}', {value_columns[i]}, "
    
    # 2) Printing (or not) the expression
    if print_stack_expr is True:
        print(stack_expr)

    # 3) Creating the pivot    
    dfs_long = dfs.select(key_columns + [F.expr(stack_expr)]).where(f"{value_column_name} is not null")
    return dfs_long

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


@typechecked
def with_start_week(df: pyspark.sql.dataframe.DataFrame , date_col: str, start_day: str = 'sunday'):
    """Function that adds a date column with the week start.

    Args:
        df (pyspark.sql.dataframe.DataFrame): Spark DataFrame.
        date_col (str): Column name of the column from which the start week date will be computed from.
            Must be one of: pyspark.sql.types.TimestampType, pyspark.sql.types.DateType.
        start_day (str, optional): Which day should the week start?. Defaults to 'sunday'.

    Raises:
        ValueError: If date_col is not from type: pyspark.sql.types.TimestampType.

    Returns:
        [pyspark.sql.dataframe.DataFrame]: The df argument with a column date column called week.
    """
    if not type(df.schema[date_col].dataType) in [pyspark.sql.types.TimestampType, pyspark.sql.types.DateType]:
        raise ValueError(f'Column {date_col} is of wrong type. See the docstring argument.')
    return df.withColumn("week", F.date_sub(F.next_day(F.col(date_col), start_day),7))
    