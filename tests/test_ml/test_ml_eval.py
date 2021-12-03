import numpy as np
import pandas as pd
import pyspark
import pytest

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

