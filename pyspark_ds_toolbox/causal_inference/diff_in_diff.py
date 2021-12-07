"""Difference in Difference toolbox.

For an introduction on the subject see: https://mixtape.scunning.com/difference-in-differences.html
"""

from typing import Union, List
from typeguard import typechecked

import pyspark
import pyspark.sql.functions as F
from pyspark.ml.regression import LinearRegression

from pyspark_ds_toolbox.ml.data_prep import get_features_vector

@typechecked
def did_estimator(
    df: pyspark.sql.dataframe.DataFrame,
    id_col: str, 
    y: str, 
    flag_unit: str, 
    flag_time: str, 
    num_features: Union[None, List[str]] = None,
    cat_features: Union[None, List[str]] = None
) -> dict:
    """Difference in Difference Estimator.

    implementation based on https://matheusfacure.github.io/python-causality-handbook/14-Difference-in-Difference.html.
    
    Args:
        df (pyspark.sql.dataframe.DataFrame): SparkDF from which the causal effect will be estimated.
        id_col (str): Column name of an unique unit identifier.
        y (str): Column name of the outcome of interest.
        flag_unit (str): Column name of a flag indicating wheter the unit was treated or not. MUST be 1 or 0.
        flag_time (str): Column name of a flag indicating whether the time is before or after the treatment. MUST BE 1 or 0.
        num_features ([type], optional): List of numerics features to be used. Defaults to Union[None, List[str]].
        cat_features ([type], optional): List of categorical features to be used. Defaults to None.

    Raises:
        ValueError: If id_col, y, flag_unit or flag_time is not in df.columns.

    Returns:
        [dict]: A dictionary with the following keys and values
            - 'impacto_medio': list(linear_model.coefficients)[0],
            - 'n_ids_impactados': df_model.select(id_col).distinct().count(),
            - 'impacto': list(linear_model.coefficients)[0]*df_model.select(id_col).distinct().count(),
            - 'pValueInteraction': linear_model.summary.pValues[0],
            - 'r2': linear_model.summary.r2,
            - 'r2adj': linear_model.summary.r2adj,
            - 'df_with_features': df_model,
            - 'linear_model': linear_model
    """

    for n in [id_col, y, flag_unit, flag_time]:
        if n not in df.columns:
            raise ValueError('id_col, y, flag_unit, flag_time must be columns of df.')

    df_model = df.withColumn('interaction', F.col(flag_unit)*F.col(flag_time))

    if (num_features is None) and (cat_features is None):
        num_features=['interaction', flag_unit, flag_time]
        cat_features=None
    elif (num_features is not None) and (cat_features is None):
        num_features=['interaction', flag_unit, flag_time] + num_features
        cat_features=None
    elif (num_features is None) and (cat_features is not None):
        num_features=['interaction', flag_unit, flag_time]
    else:
        pass

    df_model = get_features_vector(
        df=df_model,
        num_features=num_features,
        cat_features=cat_features
    )

    lin_reg = LinearRegression(featuresCol = 'features', labelCol='deposits', fitIntercept=True)
    linear_model = lin_reg.fit(df_model)


    out_dict = {
        'impacto_medio': list(linear_model.coefficients)[0],
        'n_ids_impactados': df_model.select(id_col).distinct().count(),
        'impacto': list(linear_model.coefficients)[0]*df_model.select(id_col).distinct().count(),
        'pValueInteraction': linear_model.summary.pValues[0],
        'r2': linear_model.summary.r2,
        'r2adj': linear_model.summary.r2adj,
        'df_with_features': df_model,
        'linear_model': linear_model
    }
    return out_dict
