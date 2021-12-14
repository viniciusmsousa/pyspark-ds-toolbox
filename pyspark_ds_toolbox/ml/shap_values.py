"""This modules implements a way to estimate shap values from SparkDfs.

We are not able to implement a generic function/class for pure spark model, that took in account numerical and categorical features.
So this implementation as a workaround to get the job done. The main function is estimate_shap_values() and you can check directly 
the documentation on this function. 
"""
from typing import Union, List
import pandas as pd
import pyspark
from pyspark.sql import functions as F, types as T, Window as W
import h2o
from h2o.automl import H2OAutoML


class ShapValuesPDF():
    """
    H2O AutoML Wrapper
    """
    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        target_col: str,
        cat_features: Union[None, List[str]],
        sort_metric: str,
        problem_type: str,
        max_mem_size: str = '3G',
        max_models: int = 10,
        max_runtime_secs: int = 60,
        nfolds: int = 5,
        seed: int = 90
    ):
        # Checking for values errors
        if target_col not in df.columns:
            raise ValueError(f'Column {target_col} not in df.columns.')

        if problem_type not in ['classification', 'regression']:
            raise ValueError(f'problem_type: {problem_type}. Should be in ["classification", "regression"].')


        # Params Values
        self.max_mem_size = max_mem_size
        self.df = df
        self.id_col = id_col
        self.target_col = target_col
        self.cat_features = cat_features
        self.sort_metric = sort_metric
        self.problem_type = problem_type
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed
        
        self.df_id = self.df[[self.id_col]]
        # 1) Starting H2O
        self.start_h2o()
        
        # 2) Getting Features Cols
        self.feature_cols = self.get_feature_cols()
        
        # 3) Spliting into Train and Valid
        self.h2o_df = self.as_h2o_df()

        # 4) Training Model
        self.h2o_automl = self.fit_automl()

        # 5) Shap Values
        self.shap_values = self.extract_shap_values()


    def start_h2o(self):
        h2o.init(max_mem_size=self.max_mem_size)

    def get_feature_cols(self):
        return list(set(list(self.df.columns)).difference([self.target_col, self.id_col]))

    def as_h2o_df(self):

        if self.cat_features is not None:
            for c in self.cat_features:
                h2o_df = pd.concat([self.df, pd.get_dummies(self.df[c], drop_first=True, prefix=c)], axis=1)
        else:
            h2o_df = self.df.copy()

        h2o_df = h2o.H2OFrame(h2o_df.drop(columns=[self.id_col]))
        if self.problem_type == 'classification':
            h2o_df[self.target_col] = h2o_df[self.target_col].asfactor()
        
        return h2o_df

    def fit_automl(self):
        ## 1) Training Model
        #https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
        model = H2OAutoML(
            sort_metric = self.sort_metric,
            max_models = self.max_models, 
            max_runtime_secs = self.max_runtime_secs,
            nfolds = self.nfolds,
            seed = self.seed,
            include_algos = ["GBM", "XGBoost", "DRF"] # models that have shap values.
        )
        model.train(
            x = self.feature_cols,
            y = self.target_col, 
            training_frame = self.h2o_df,
        )

        return model
    
    def extract_shap_values(self):
        fi = self.h2o_automl.leader.predict_contributions(test_data=self.h2o_df).as_data_frame()
        _features = list(fi.columns)
        _features.remove('BiasTerm')

        # Proportional weights
        fi['MeanDeviance'] = fi.drop(columns=['BiasTerm']).sum(axis=1)
        fi = pd.concat([fi[_features].div(fi['MeanDeviance'], axis=0), fi[['MeanDeviance', 'BiasTerm']]], axis=1)
        fi.drop(columns=['MeanDeviance', 'BiasTerm'], inplace=True)
        
        # Wide to Long
        fi = pd.concat([self.df_id.reset_index(drop=True), fi.reset_index(drop=True)], axis=1)
        fi = pd.melt(frame=fi, id_vars=[self.id_col], ignore_index=True, value_name='shap_value')
        return fi

def estimate_shap_values(
    sdf: pyspark.sql.dataframe.DataFrame,
    id_col: str,
    target_col: str,
    cat_features: Union[List[str], None],
    sort_metric: str,
    problem_type: str,
    subset_size: int = 2000,
    max_mem_size: str = '2G',
    max_models: int = 8,
    max_runtime_secs: int = 30,
    nfolds: int = 5,
    seed: int =90
):
    """Computes for each row the shap values of each feature.

    This function will split the sdf into int(sdf.count()/subset_size) pandas dataframes and then use the 
    Class ShapValuesPDF, which is a wrapper of the H2O automl, on each the subseted dataset using the 
    applyInPandas() method of grouped SparkDF.
    
    Check the following link for an intuition of how this works:
    https://www.youtube.com/watch?v=x6dSsbXhyPo 

    Args:
        sdf (pyspark.sql.dataframe.DataFrame): A SparkDF. Note that all column, except the id_col must be features. 
        id_col (str): Column name of the identifier.
        target_col (str): Column name of the target value.
        cat_features (Union[List[str], None]): List of column names of the categorical variables, if any.
        sort_metric (str): A metric to sort the candidates (see https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/sort_metric.html).
        problem_type (str): 'regression' or 'classification'. If classification then target_col must be of binary values.
        subset_size (int, optional): Number of rows for each sub dataset. Defaults to 2000.
        max_mem_size (str, optional): Max memory size to be allocated to h2o local cluster. Defaults to '2G'. (see https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/h2o.html#h2o.init)
        max_models (int, optional): Max number of model to be fitted. These models are ranked according to sort_metric. Defaults to 8. (see https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html?highlight=automl#h2oautoml)
        max_runtime_secs (int, optional): Max number of seconds to spend fitting the models. Defaults to 30. (see https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html?highlight=automl#h2oautoml)
        nfolds (int, optional): Number of folds to be used for cross validation while fitting. Defaults to 5. (see https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html?highlight=automl#h2oautoml)
        seed (int, optional): Seed. Defaults to 90.

    Raises:
        ValueError: if int(sdf.count()/subset_size) < 2 is True

    Returns:
        [pyspark.sql.dataframe.DataFrame]: A sparkDF with the follwing columns
            - id_col: The values from the column passed as id_col;
            - feature: The name of each feature from features_names;
            - shap_value: The shap value of the feature.
    """
    schema = T.StructType([
      T.StructField(id_col, sdf.schema[id_col].dataType),
      T.StructField('variable', T.StringType()),
      T.StructField('shap_value', T.FloatType()),
    ])

    n_quantiles = int(sdf.count()/subset_size)
    if n_quantiles < 2:
        raise ValueError('subset_size must be smaller to result in at least two subsets from sdf. The condition int(sdf.count()/subset_size) < 2 was True')
    sdf = sdf\
        .withColumn('rand', F.rand())\
        .withColumn('qcut', F.ntile(n_quantiles).over(W.partitionBy().orderBy(F.col('rand'))))\
        .withColumn('qcut', F.col('qcut').cast(T.StringType()))

    def __fn(pdf):
        model = ShapValuesPDF(
            df=pdf,
            max_mem_size=max_mem_size,
            id_col=id_col,
            target_col=target_col,
            cat_features=cat_features,
            sort_metric=sort_metric,
            problem_type=problem_type,
            max_models=max_models,
            max_runtime_secs=max_runtime_secs,
            nfolds=nfolds,
            seed=seed
        )
        return model.shap_values

    return sdf.groupBy('qcut').applyInPandas(__fn, schema=schema)
