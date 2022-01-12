"""Module dedicated to features spark vector tools.
"""

from typing import List, Union
from typeguard import typechecked

import pyspark 
import pyspark.ml.feature as FF
from pyspark.ml import Pipeline


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
