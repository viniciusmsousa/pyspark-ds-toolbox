"""Module dedicated to functionalities related to class weighting tools.
"""

from typeguard import typechecked

import pyspark 
from pyspark.sql import functions as F


@typechecked
def binary_classifier_weights(dfs: pyspark.sql.dataframe.DataFrame, col_target: str) -> pyspark.sql.dataframe.DataFrame:
    """Adds a class weight columns to a binary classification response column.

    Args:
        dfs (pyspark.sql.dataframe.DataFrame): Training dataset with the col_target column.
        col_target (str): Column name of the column that contains the response variable for the model.
            It should contain only values of 0 and 1.


    Raises:
        ValueError: If unique values from col_target column are not 0 and 1.

    Returns:
        pyspark.sql.dataframe.DataFrame: The dfs object with a weight_{col_target} column.
    """
    # 1) Check if col_target unique values are [0, 1] 
    unique_rows = dfs.select(col_target).distinct().collect()
    unique_rows_values = [v[col_target] for v in unique_rows]
    unique_rows_values.sort()
    if unique_rows_values != [0, 1]:
        raise ValueError(f'Unique values from {col_target} column are not [0, 1].')
    
    # 2) Computes the class weights
    count_parc = dfs.filter(f'{col_target} = 1').count()
    count_total = dfs.count()
    c = 2
    weight_parc = count_total / (c * count_parc)
    weight_no_parc = count_total / (c * (count_total - count_parc)) 

    # 3) Adds the weight column to dfs
    df_with_weight_columns = dfs.withColumn(f"weight_{col_target}", F.when(F.col(col_target) == 1, weight_parc).otherwise(weight_no_parc))

    return df_with_weight_columns
