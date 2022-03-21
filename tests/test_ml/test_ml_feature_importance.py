import pandas as pd
import pyspark
from pyspark.sql import types as T, functions as F
import pyspark.ml.classification as spark_cl
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import pytest
from pyspark_ds_toolbox.ml.data_prep import get_features_vector

from pyspark_ds_toolbox.ml.feature_importance.shap_values import estimate_shap_values
import pyspark_ds_toolbox.ml.feature_importance.native_spark as sparktb_fi


def test_estimate_shap_values(df_causal_inference):

    shap_values = estimate_shap_values(
        sdf=df_causal_inference.sample(fraction=0.1, withReplacement=False),
        id_col='index',
        target_col='re78',
        cat_features = ['data_id'],
        sort_metric='rmse',
        problem_type='regression',
        subset_size = 500,
        max_mem_size = '2G',
        max_models=8,
        max_runtime_secs=15,
        nfolds=5,
        seed=90
    )

    assert type(shap_values) == pyspark.sql.dataframe.DataFrame
    
    s = T.StructType([
      T.StructField('index', df_causal_inference.schema['index'].dataType),
      T.StructField('variable', T.StringType()),
      T.StructField('shap_value', T.FloatType()),
    ])
    assert shap_values.schema == s

def test_native_spark_extract_features_score(df_causal_inference):
  # Testing the functionalities
  df_causal_inference = df_causal_inference\
        .withColumn('etnia', F.expr('case when black=1 then "black" when hisp=1 then "hisp" when marr=1 then "marr" else "other" end'))\
        .drop('black', 'hisp', 'marr', 'features', 'num', 'cat')
  
  num_features = ['age', 'educ', 'nodegree', 're74', 're75', 're78', 'age2', 'age3', 'educ2', 'educ_re74', 'u74', 'u75']
  cat_features = ['etnia']
  stages = get_features_vector(num_features=num_features, cat_features=cat_features)
   
  # Testing Gini Score
  dt = Pipeline(stages = stages+[spark_cl.DecisionTreeClassifier(labelCol='treat', featuresCol='features')]).fit(df_causal_inference)
  df_fi = sparktb_fi.extract_features_score(model=dt.stages[-1], dfs=dt.transform(df_causal_inference), features_col='features')
  assert type(df_fi) == pd.core.frame.DataFrame
  assert df_fi.shape == (15, 3)
  assert list(df_fi.columns) == ['feat_index', 'feature', 'delta_gini']

  # Testing Odds Ratio
  lr = Pipeline(stages = stages+[spark_cl.LogisticRegression(labelCol='treat', featuresCol='features')]).fit(df_causal_inference)
  df_fi = sparktb_fi.extract_features_score(model=lr.stages[-1], dfs=lr.transform(df_causal_inference), features_col='features')
  assert type(df_fi) == pd.core.frame.DataFrame
  assert df_fi.shape == (15, 3)
  assert list(df_fi.columns) == ['feat_index', 'feature', 'odds_ratio']

  # Testing Coefficients
  linearr = Pipeline(stages = stages+[LinearRegression(labelCol='re78', featuresCol='features')]).fit(df_causal_inference)
  df_fi = sparktb_fi.extract_features_score(model=linearr.stages[-1], dfs=linearr.transform(df_causal_inference), features_col='features')
  assert type(df_fi) == pd.core.frame.DataFrame
  assert df_fi.shape == (15, 3)
  assert list(df_fi.columns) == ['feat_index', 'feature', 'coefficients']

  # Testing Raise Value errors
  with pytest.raises(ValueError):
      sparktb_fi.extract_features_score(model=dt.stages[-1], dfs=df_causal_inference, features_col='aaaaaaa')