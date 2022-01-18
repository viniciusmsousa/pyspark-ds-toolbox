"""Module with spark native feature importance score tools.
"""
from typing import Union
from typeguard import typechecked
from numpy import exp as np_exp
import pandas as pd

import pyspark


@typechecked
def extract_features_score(
    model: Union[
        pyspark.ml.classification.LogisticRegressionModel,
        pyspark.ml.classification.DecisionTreeClassificationModel,
        pyspark.ml.classification.RandomForestClassificationModel,
        pyspark.ml.classification.GBTClassificationModel,
        pyspark.ml.regression.LinearRegressionModel,
        pyspark.ml.regression.DecisionTreeRegressionModel,
        pyspark.ml.regression.RandomForestRegressionModel,
        pyspark.ml.regression.GBTRegressionModel
    ],
    dfs: pyspark.sql.dataframe.DataFrame,
    features_col: str = 'features'
) -> pd.core.frame.DataFrame:
    """Function that extracts feature importance or coefficients from spark models.

    There are 3 possible situations created inside this function:
        1) If model is a DecisionTree, RandomForest or GBT (classification or regression) the features score will be gini scores.
        See https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassificationModel.html?highlight=featureimportances#pyspark.ml.classification.DecisionTreeClassificationModel.featureImportances
        for a description on the feature importance;
        
        2) If model is a LogisticRegression the features scores will be the odds ratio;

        3) If model is a LinearRegression the features socres will be the coefficients.

    Args:
        model (Union[spark models]): A fitted spark model. Accepted models: DecisionTree, RandomForest, GBT (both classification and regression), LogisticRegression and LinearRegression.
        dfs (pyspark.sql.dataframe.DataFrame): The SparkDataFrame used to fit the model.
        features_col (str, optional): The features vector column. Defaults to 'features'.

    Raises:
        ValueError: if features_col not in dfs.columns is True.

    Returns:
        [pd.core.frame.DataFrame]: A PandasDataFrame with the folloing columns:
            - feat_index: The feature index number in the features vector column;
            - feature: The feature name;
            - delta_gini/odds_ratio/coefficient: The feature score depending on the model (see the description).
    """
    if features_col not in dfs.columns:
        raise ValueError(f'features_col "{features_col}" not in dfs.columns.')

    list_extract = []
    # If regression then get coefficients else get featureImportances (gini)
    if type(model) in (pyspark.ml.classification.LogisticRegressionModel, pyspark.ml.regression.LinearRegressionModel):
        featureImp = model.coefficients
    else:
        featureImp = model.featureImportances

    for i in dfs.schema[features_col].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dfs.schema[features_col].metadata["ml_attr"]["attrs"][i]
    
    df_fi = pd.DataFrame(list_extract)
    df_fi['score'] = df_fi['idx'].apply(lambda x: featureImp[x])
    df_fi = df_fi.sort_values('score', ascending = False)
    if 'vals' in list(df_fi.columns):
        df_fi.drop(columns=['vals'], inplace=True)

    if type(model) == pyspark.ml.classification.LogisticRegressionModel:
        df_fi.columns = ['feat_index', 'feature', 'odds_ratio']
        df_fi['odds_ratio'] = np_exp(df_fi['odds_ratio'])
    
    elif type(model) == pyspark.ml.regression.LinearRegressionModel:
        df_fi.columns = ['feat_index', 'feature', 'coefficients']
    
    else:
        df_fi.columns = ['feat_index', 'feature', 'delta_gini']
    
    return df_fi
