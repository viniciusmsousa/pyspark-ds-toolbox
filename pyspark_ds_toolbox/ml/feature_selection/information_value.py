from multiprocessing.sharedctypes import Value
from typing import Tuple, List, Union
from warnings import WarningMessage
from typeguard import typechecked
import pandas as pd

from pyspark import keyword_only
from pyspark.sql import DataFrame, functions as F, types as T, SparkSession
from pyspark.ml import Transformer
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark_ds_toolbox.ml.data_prep.features_vector import get_features_vector

@typechecked
def compute_woe_iv(
    dfs: DataFrame,
    col_feature: str,
    col_target: str
) -> Tuple[DataFrame, float]:
    """Function that given a DataFrame, a categorical feature and a binary target column computes the Information Value.
    
    See http://www.m-hikari.com/ams/ams-2014/ams-65-68-2014/zengAMS65-68-2014.pdf for a technical description.

    Args:
        dfs (DataFrame): A spark DataFrame with col_feature and col_target;
        col_feature (str): Column name of a categorical feature.
        col_target (str): Column name of a binary target column. Must be of integer type and have only value 0 and 1.

    Raises:
        TypeError: if dfs.schema[col_target].dataType != T.IntegerType
        ValueError: unique_target_values != [0, 1]

    Returns:
        Tuple[DataFrame, float]: A two element tuple with the follwing objects:
            - Spark DataFrame with 'feature', 'feature_value', 'woe', 'iv' column;
            - float with the col_feature information value.
    """
    if dfs.schema[col_target].dataType != T.IntegerType():
        raise TypeError(f'Column {col_target} is of type {dfs.schema[col_target].dataType}. Must be IntegerType.')
    
   
    unique_target_values = dfs.select(col_target).distinct().collect()
    unique_target_values = [v[col_target] for v in unique_target_values]
    unique_target_values.sort()    
    if unique_target_values != [0, 1]:
        raise ValueError(f'Unique values from {col_target} column are not [0, 1].')
    
     
    cross = dfs\
        .crosstab(col_feature, col_target)\
        .withColumnRenamed(f'{col_feature}_{col_target}', 'feature_value')
        
    sum_0 = cross.select('0').groupBy().sum().collect()[0][0]
    sum_1 = cross.select('1').groupBy().sum().collect()[0][0]
    
    cross = cross\
        .withColumn('0', F.col('0')/sum_0)\
        .withColumn('1', F.col('1')/sum_1)\
        .withColumn('woe', F.log(F.col('0')/F.col('1')))\
        .withColumn('iv', F.col('woe')*(F.col('0') - F.col('1')))\
        .withColumn('feature', F.lit(col_feature))\
        .fillna(0)

    iv = cross.selectExpr('sum(iv) as iv').collect()[0][0]  
        
    return cross.select('feature', 'feature_value', 'woe', 'iv'), iv


class WeightOfEvidenceComputer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasInputCols,
    HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable
):
    """A transform to add the weight of evidence value for categorical variables.
    
    This is a class to add the columns with the Weight of Evidence for categorical features.
    See http://www.m-hikari.com/ams/ams-2014/ams-65-68-2014/zengAMS65-68-2014.pdf for a technical description.  
    """
    
    col_target = Param(
        parent=Params._dummy(),
        name='col_target',
        doc='Column name of the target. Must be a integer os values 0 or 1.',
        typeConverter=TypeConverters.toString
    )

    @keyword_only
    def __init__(
        self,
        inputCol=None,
        inputCols=None,
        col_target=None,
    ):
        super().__init__()
        self._setDefault(col_target=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol=None,
        inputCols=None,
        col_target=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setTarget(self, new_target):
        return self.setParams(col_target=new_target)

    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)

    def setInputCols(self, new_inputCols):
        return self.setParams(inputCols=new_inputCols)

    def getTarget(self):
        return self.getOrDefault(self.col_target)
    
    def checkParams(self):
        # Test #1: either inputCol or inputCols can be set (but not both).
        if self.isSet("inputCol") and (self.isSet("inputCols")):
            raise ValueError("Only one of `inputCol` and `inputCols`" "must be set.")

        # Test #2: at least one of inputCol or inputCols must be set.
        if not (self.isSet("inputCol") or self.isSet("inputCols")):
            raise ValueError("One of `inputCol` or `inputCols` must be set.")
    
    @typechecked
    def add_woe(self, dfs: DataFrame, feature: str, col_target: str) -> DataFrame:
        """Function that add teh WOE to the dataset passed to the tranform method of the class.

        Args:
            dfs (DataFrame): A spark DataFrame with feature and target columns.
            feature (str): Column name of a categorical feature, must be of type string.
            col_target (str): Column name of a target variable, must be of type integer with values 0 and 1.

        Returns:
            DataFrame: The dfs argument with a column f'{feature}_woe'.
        """

        dfs_woe_feature, iv = compute_woe_iv(dfs=dfs,col_feature=feature,col_target=col_target)
        dfs_woe_feature = dfs_woe_feature\
            .withColumnRenamed('feature_value', feature)\
            .withColumnRenamed('woe', f'{feature}_woe')\
            .select(feature, f'{feature}_woe')


        cols = dfs.columns + [f'{feature}_woe']
        dfs = dfs.join(dfs_woe_feature, on=feature, how='left').select(cols)
        return dfs

    def _transform(self, dataset):
        self.checkParams()

        # If `inputCol`, we wrap into a single-item list
        input_columns = (
            [self.getInputCol()]
            if self.isSet("inputCol")
            else self.getInputCols()
        )

        for feat in input_columns:
            dataset = self.add_woe(dfs=dataset, feature=feat, col_target=self.getTarget())
            
        return dataset

@typechecked
def feature_selection_with_iv(
    dfs: DataFrame,
    col_target: str,
    num_features: Union[List[str], None],
    cat_features: Union[List[str], None],
    floor_iv: float = 0.3,
    bucket_fraction: float = 0.1,
    categorical_as_woe: Union[bool, None] = False
) -> dict:
    """Function that executes a feature selection based on the information value methodology.

    This function computes the information value for the features passed and based on the floor_iv select the features
    that should ne used in the modeling part. It return analytical dataframes (with the Weight of Evidence and Information
    Value). 

    But the main advantage is that it also returns a list os stages to be passed to a pyspark.ml.Pipeline that will encode 
    and assemble the select variables.


    See http://www.m-hikari.com/ams/ams-2014/ams-65-68-2014/zengAMS65-68-2014.pdf for a technical description
    and https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html for a introduction on the subject.

    Args:
        dfs (DataFrame): A spark DataFrame.
        col_target (str): Column name of a binary target column. Must be of integer type and have only value 0 and 1.
        num_features (Union[List[str], None]): List of columns names of numeric features. This column will be transformed with QuantileDiscretizer.
        cat_features (Union[List[str], None]): List of columns names of categorical features;
        floor_iv (float, optional): Threshold for a feature to be selected, greater or equal this value. Defaults to 0.3.
        bucket_fraction (float, optional): Fraction of the dataset to be used to create the buckets Must be between 0.05 and 0.5. Defaults to 0.1. 
        categorical_as_woe (bool, optional): If True categorical variables will as Weight of Evidence. If False one hot encoding. Defaults to False.

    Raises:
        TypeError: if (num_features is None) and (cat_features is None)
        ValueError: if (bucket_fraction < 0.05) or (bucket_fraction > 0.5)

    Returns:
        dict: A dict with the following structure
            - dfs_woe: Spark DataFrame with WOE and IV for each feature value. Given by the cols: feature, feature_value, woe, iv.
            - dfs_iv: Spark DataFrame with IV for each feature. Given by the cols: feature, iv.
            - stages_features_vector: List with spark transformers that computes the features vector based on the floor_iv and categorical_as_woe params.
    """

    if (num_features is None) and (cat_features is None):
        raise TypeError('num_features or cat_features must be a List[str]. Both are none.')
    
    if (bucket_fraction < 0.05) or (bucket_fraction > 0.5):
        raise ValueError('Param floor_bucket_fraction must be between 0.05 and 0.5')

    if (type(categorical_as_woe) is bool) and (cat_features is None):
        raise Warning('cat_features is None and categorical_as_woe is bool. categorical_as_woe param will have no effect.')

    if num_features is not None:
        count_dfs = dfs.count()
        nBuckets = count_dfs/(count_dfs*bucket_fraction)

        bucket_num_features = [i + '_bucket' for i in num_features]
        qt = QuantileDiscretizer(inputCols=num_features, outputCols=bucket_num_features, numBuckets=nBuckets)
        dfs = qt.fit(dfs).transform(dfs)
    
    if (num_features is not None) and (cat_features is not None):
        feats = bucket_num_features + cat_features
    elif num_features is None:
        feats = cat_features
    else:
        feats = bucket_num_features


    sc = SparkSession.builder.appName('spark_session_getter').getOrCreate()
    schema_woe = T.StructType([
        T.StructField("feature", T.StringType(), False),
        T.StructField("feature_value", T.StringType(), True),
        T.StructField("woe", T.FloatType(), True),
        T.StructField("iv", T.FloatType(), True)
    ])
    dfs_woe = sc.createDataFrame([], schema_woe)


    schema_iv = T.StructType([
        T.StructField('feature', T.StringType(), False),
        T.StructField('iv', T.FloatType(), False)
    ])
    dfs_iv = sc.createDataFrame([], schema_iv)

    for f in feats:
        df_woe_feature, iv = compute_woe_iv(dfs=dfs,col_feature=f,col_target=col_target)
    
        dfs_woe = dfs_woe.union(df_woe_feature)
        dfs_iv = dfs_iv.union(sc.createDataFrame(pd.DataFrame({'feature':[f],'iv':[iv]})))

    
    cols_to_keep = dfs_iv.filter(f'iv >= {floor_iv}').toPandas()['feature'].to_list()

    if (cat_features is not None) and (categorical_as_woe==True):
        cat_features_selected = list(filter(None, [None if s.endswith('_bucket') else s for s in cols_to_keep]))
        selected_features = [s[:-7] if s.endswith('_bucket') else s+'_woe' for s in cols_to_keep]

        stages_features_vector = [WeightOfEvidenceComputer(inputCols=cat_features_selected, col_target=col_target)] \
            + get_features_vector(num_features=selected_features)
    else:
        num_selectec_features = list(filter(None, [s[:-7] if s.endswith('_bucket') else None for s in cols_to_keep]))
        cat_selectec_features = list(filter(None, [None if s.endswith('_bucket') else s for s in cols_to_keep]))
        
        stages_features_vector = get_features_vector(num_features=num_selectec_features, cat_features=cat_selectec_features)

    out_dict = {
        'dfs_woe': dfs_woe,
        'dfs_iv': dfs_iv.orderBy(F.col('iv').desc()),
        'stages_features_vector': stages_features_vector
    }

    return out_dict
