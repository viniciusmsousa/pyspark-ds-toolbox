import numpy as np
import pandas as pd
import pyspark
import pytest

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier

from pyspark_ds_toolbox.ml.data_prep import get_p1
from pyspark_ds_toolbox.ml import eval as ml_ev 


def test_binary_classificator_evaluator(dfs_binary_classificator_evaluator):
    out = ml_ev.binary_classificator_evaluator(
        dfs_prediction=dfs_binary_classificator_evaluator,
        col_target='target',
        col_prediction='predicted'
    )

    for metric in ['confusion_matrix', 'accuracy', 'f1', 'precision', 'recall', 'aucroc', 'aucpr']:
        assert metric in out.keys()

    assert type(out['confusion_matrix']) == pd.DataFrame
    assert type(out['accuracy']) in [float, np.nan]
    assert type(out['f1']) in [float, np.nan]
    assert type(out['precision']) in [float, np.nan]
    assert type(out['recall']) in [float, np.nan]
    assert type(out['aucroc']) in [float, np.nan]
    assert type(out['aucpr']) in [float, np.nan]

def test_binary_classifier_decile_analysis(dfs_decile_analysis_input, dfs_decile_analysis_output):
    decile_table = ml_ev.binary_classifier_decile_analysis(
        dfs=dfs_decile_analysis_input,
        col_id='id_conta',
        col_target='target_value',
        col_probability='p1'
    )
    pd.testing.assert_frame_equal(decile_table.toPandas(), dfs_decile_analysis_output)

def test_binary_classifier_decile_analysis_error(spark):
    dfs = spark.createDataFrame(pd.DataFrame.from_dict({
        'index': ['a',' b', 'c'],
        'target': [0, 1, 2],
        'probability': [0.1, 0.2, 0.3]
    }))
    with pytest.raises(ValueError):
        ml_ev.binary_classifier_decile_analysis(
            dfs=dfs,
            col_id='index',
            col_target='target',
            col_probability='probability'
        )

def test_estimate_individual_shapley_values(spark, input_estimate_individual_shapley_values):
    train =  input_estimate_individual_shapley_values[0]
    row_of_interest = input_estimate_individual_shapley_values[2]
    df_causal_inference = input_estimate_individual_shapley_values[3]
    features=['age', 'age2', 'age3', 'educ', 'educ2', 'marr', 'nodegree', 'black', 'hisp', 're74', 're75', 'u74', 'u75', 'educ_re74']

    # Regression
    model_regressor = GBTRegressor(labelCol='re78')
    p_regression = Pipeline(stages=[model_regressor]).fit(train)
    df_pred = p_regression.transform(df_causal_inference)

    sdf_shap_regression = ml_ev.estimate_individual_shapley_values(
        spark=spark,
        df = df_pred,
        id_col='index',
        model = p_regression,
        column_of_interest='prediction',
        problem_type='regression',
        row_of_interest = df_pred.first(),
        feature_names = features,
        features_col='features',
        print_shap_values=False
    )
    assert type(sdf_shap_regression) == pyspark.sql.dataframe.DataFrame
    assert sdf_shap_regression.count() == len(features)
    assert sdf_shap_regression.columns == ['index', 'feature', 'shap']

    # Classification
    model_classification = GBTClassifier(labelCol='treat')
    p_classification = Pipeline(stages=[model_classification]).fit(train)
    df_pred = p_classification.transform(df_causal_inference)\
        .withColumn('p1', get_p1(F.col('probability')))

    sdf_shap_classification = ml_ev.estimate_individual_shapley_values(
        spark=spark,
        df = df_pred,
        id_col='index',
        model = p_classification,
        column_of_interest='p1',
        problem_type='classification',
        row_of_interest = df_pred.first(),
        feature_names = features,
        features_col='features',
        print_shap_values=False
    )
    assert type(sdf_shap_classification) == pyspark.sql.dataframe.DataFrame
    assert sdf_shap_classification.count() == len(features)
    assert sdf_shap_classification.columns == ['index', 'feature', 'shap']

    # Errors
    with pytest.raises(ValueError):
        ml_ev.estimate_individual_shapley_values(
            spark=spark,
            df = df_causal_inference.withColumn('a', F.lit('A')),
            id_col='a',
            model = p_regression,
            column_of_interest='prediction',
            problem_type='regression',
            row_of_interest = row_of_interest,
            feature_names = features,
            features_col='features',
            print_shap_values=False
        )
    
    with pytest.raises(ValueError):
        ml_ev.estimate_individual_shapley_values(
            spark=spark,
            df = df_causal_inference,
            id_col='index',
            model = p_regression,
            column_of_interest='prediction',
            problem_type='regression',
            row_of_interest = row_of_interest,
            feature_names = features + ['a'],
            features_col='features',
            print_shap_values=False
        )
