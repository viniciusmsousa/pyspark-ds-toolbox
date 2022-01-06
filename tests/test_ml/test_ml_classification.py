import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from pyspark_ds_toolbox.ml.classification import eval as cl_ev 
from pyspark_ds_toolbox.ml.classification.baseline_classifiers import baseline_binary_classfiers

def test_binary_classificator_evaluator(dfs_binary_classificator_evaluator):
    out = cl_ev.binary_classificator_evaluator(
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
    decile_table = cl_ev.binary_classifier_decile_analysis(
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
        cl_ev.binary_classifier_decile_analysis(
            dfs=dfs,
            col_id='index',
            col_target='target',
            col_probability='probability'
        )

def test_baseline_classifiers(df_causal_inference):
    df_causal_inference = df_causal_inference\
        .withColumn('etnia', F.expr('case when black=1 then "black" when hisp=1 then "hisp" when marr=1 then "marr" else "other" end'))\
        .drop('black', 'hisp', 'marr', 'features')

    num_features = ['age', 'educ', 'nodegree', 're74', 're75', 're78', 'age2', 'age3', 'educ2', 'educ_re74', 'u74', 'u75']
    cat_features = ['etnia']
    target_col = 'treat' # must be 1 or 0
    dfs_train, dfs_test = df_causal_inference.randomSplit([0.8, 0.2], seed=4)

    t = baseline_binary_classfiers(
        dfs=dfs_train,
        id_col='index',
        target_col=target_col,
        num_features=num_features,
        cat_features=cat_features,
        dfs_test=dfs_test,
        weight_on_target=True,
        log_mlflow_run=False,
        artifact_stage_path = None
    )
    assert type(t) == dict
    main_keys = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GBTClassifier']
    assert list(t.keys()) == main_keys

    for k in main_keys:
        t[k] == ['model', 'metrics', 'decile_metrics']

    t = baseline_binary_classfiers(
        dfs=dfs_train,
        id_col='index',
        target_col=target_col,
        num_features=num_features,
        cat_features=cat_features,
        dfs_test=dfs_test,
        weight_on_target=False,
        log_mlflow_run=False,
        artifact_stage_path = None
    )
    assert type(t) == dict
    main_keys = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GBTClassifier']
    assert list(t.keys()) == main_keys

    for k in main_keys:
        t[k] == ['model', 'metrics', 'decile_metrics']
