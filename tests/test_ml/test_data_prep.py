import pytest
import pandas as pd
import pyspark
from pyspark.ml import Pipeline

from pyspark_ds_toolbox.ml import data_prep as ml_dp

def test_get_features_vector(df_spark_features_col, schema_get_features_vector):

    stages = ml_dp.features_vector.get_features_vector(
        num_features=['num1', 'num2'],
        cat_features=['cat1', 'cat2']
    )
    d = Pipeline(stages=stages).fit(df_spark_features_col).transform(df_spark_features_col)
    
    for i in [0, 1, 2, 3, 4]:
        assert list(d.schema)[i] == list(schema_get_features_vector)[i]

def test_get_features_vector_type_error(df_spark_features_col):
    with pytest.raises(TypeError):
        ml_dp.features_vector.get_features_vector(
            df=df_spark_features_col,
            num_features=None,
            cat_features=None
        )

def test_binary_classifier_weights(dfs_decile_analysis_input):
    dfs_weights = ml_dp.class_weights.binary_classifier_weights(dfs=dfs_decile_analysis_input, col_target='target_value')

    assert type(dfs_weights) == pyspark.sql.dataframe.DataFrame
    assert dfs_decile_analysis_input.columns + ['weight_target_value'] == dfs_weights.columns
    assert (dfs_decile_analysis_input.count(), (len(dfs_decile_analysis_input.columns)+1)) == (dfs_weights.count(), len(dfs_weights.columns))

def test_binary_classifier_weights_error(spark):
    dfs = spark.createDataFrame(pd.DataFrame.from_dict({
        'index': ['a',' b', 'c'],
        'target': [0, 1, 2],
        'probability': [0.1, 0.2, 0.3]
    }))
    with pytest.raises(ValueError):
        ml_dp.class_weights.binary_classifier_weights(dfs=dfs, col_target='target')
