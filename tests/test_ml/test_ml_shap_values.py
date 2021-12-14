import pyspark
from pyspark.sql import types as T
from pyspark_ds_toolbox.ml.shap_values import estimate_shap_values


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
