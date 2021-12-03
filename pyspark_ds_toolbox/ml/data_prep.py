"""Machine Learning (with spark) Data Preparation toolbox.

Module dedicated to functionalities related to data preparation for ML modeling.
"""

from typing import List, Union

from typeguard import typechecked

import pyspark 
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pyspark.ml.feature as FF
from pyspark.ml import Pipeline

get_p1 = F.udf(lambda value: value[1].item(), FloatType())

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

@typechecked
def get_features_vector(
    df: pyspark.sql.dataframe.DataFrame,
    num_features: Union[List[str], None] = None,
    cat_features: Union[List[str], None] = None
) -> pyspark.sql.dataframe.DataFrame:
    """Assembles a features vector to be used with ML algorithms.

    Args:
        df (pyspark.dataframe.DataFrame): SparkDF with features to be assembled.
        num_features (List[str]): List of columns names of numeric features.
        cat_features (List[str]): List of column names of categorical features (StringIndexer).

    Raises:
        TypeError: If num_features AND cat_features are os type None.

    Returns:
        [pyspark.sql.dataframe.DataFrame]: The SparkDF passed as df with the features column.
    """
    if (num_features is None) and (cat_features is None):
        raise TypeError('num_features or cat_features must be a List[str]. Both are none.')
    
    df_cols = df.columns
    if cat_features is not None:
        indexers = [FF.StringIndexer(inputCol = c, outputCol="{0}_indexed".format(c)) for c in cat_features]
        encoders = [FF.StringIndexer(inputCol = indexer.getOutputCol(), outputCol = "{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]
        assemblerCat = FF.VectorAssembler(inputCols = [encoder.getOutputCol() for encoder in encoders], outputCol = "cat")
        pipelineCat = Pipeline(stages = indexers + encoders + [assemblerCat])
        df = pipelineCat.fit(df).transform(df)
    
    if num_features is not None:
        assemblerNum = FF.VectorAssembler(inputCols = num_features, outputCol = "num")
        pipelineNum = Pipeline(stages = [assemblerNum])
        df = pipelineNum.fit(df).transform(df)

    if (num_features is not None) and (cat_features is not None):
        columns = ['num', 'cat']
    elif num_features is None:
        columns = ['cat']
    else:
        columns = ['num']
    
    assembler = FF.VectorAssembler(inputCols = columns, outputCol = "features")
    pipeline = Pipeline(stages = [assembler])
    df_assembled = pipeline.fit(df).transform(df)

    return df_assembled.select([*df_cols,'features'])
