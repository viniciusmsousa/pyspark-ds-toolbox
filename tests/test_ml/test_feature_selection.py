import pytest
from pyspark.sql DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from pyspark_ds_toolbox.ml.feature_selection.information_value import compute_woe_iv, WeightOfEvidenceComputer, feature_selection_with_iv

# Information Value
def test_compute_woe_iv(df_causal_inference_iv):
    schema = StructType([
        StructField('feature',StringType(),False),
        StructField('feature_value',StringType(),True),
        StructField('woe',DoubleType(),False),
        StructField('iv',DoubleType(),False)
    ])
    

    df_woe_iv, iv = compute_woe_iv(dfs=df_causal_inference_iv, col_feature='etnia', col_target='treat')

    assert (type(df_woe_iv) == DataFrame) and (type(iv) == float)
    assert df_woe_iv.schema == schema
    assert round(iv, 4) == round(4.03049586578048, 4)

def test_compute_woe_iv_raises(df_causal_inference_iv):    
    with pytest.raises(TypeError):
        compute_woe_iv(dfs=df_causal_inference_iv, col_feature='etnia', col_target='etnia')

def test_WOEComputer(df_causal_inference_iv):
    cat_features = ['etnia', 'dumb_cat']
    test_woe = WeightOfEvidenceComputer(inputCols=cat_features, col_target='treat')\
        .transform(df_causal_inference_iv)

    assert type(test_woe) == type(df_causal_inference_iv)
    assert test_woe.count() == df_causal_inference_iv.count()
    assert len(test_woe.columns) == (len(df_causal_inference_iv.columns)+len(cat_features))

def test_feature_selection_with_iv(df_causal_inference_iv):

    out = feature_selection_with_iv(
        dfs=df_causal_inference_iv,
        col_target='treat',
        num_features=['age', 'educ'],
        cat_features=['etnia', 'dumb_cat'],
        floor_iv=0.3,
        bucket_fraction=0.1,
        categorical_as_woe=False
    )
    assert type(out) == dict
    assert list(out.keys()) == ['dfs_woe', 'dfs_iv', 'stages_features_vector']
    assert type(out['dfs_woe']) == DataFrame
    assert type(out['dfs_iv']) == DataFrame
    assert type(out['stages_features_vector']) == list
    assert len(out['stages_features_vector']) == 5

    out = feature_selection_with_iv(
        dfs=df_causal_inference_iv,
        col_target='treat',
        num_features=None,
        cat_features=['etnia', 'dumb_cat'],
        floor_iv=0.3,
        bucket_fraction=0.1,
        categorical_as_woe=True
    )
    assert type(out) == dict
    assert list(out.keys()) == ['dfs_woe', 'dfs_iv', 'stages_features_vector']
    assert type(out['dfs_woe']) == DataFrame
    assert type(out['dfs_iv']) == DataFrame
    assert type(out['stages_features_vector']) == list
    assert len(out['stages_features_vector']) == 3

def test_feature_selection_with_iv_raises(df_causal_inference_iv):
    with pytest.raises(ValueError):
        feature_selection_with_iv(
            dfs=df_causal_inference_iv,
            col_target='treat',
            num_features=['age', 'educ'],
            cat_features=['etnia', 'dumb_cat'],
            floor_iv=0.3,
            bucket_fraction=0.6,
            categorical_as_woe=False
        )

    with pytest.raises(TypeError):
        feature_selection_with_iv(
            dfs=df_causal_inference_iv,
            col_target='treat',
            num_features=None,
            cat_features=None,
            floor_iv=0.3,
            bucket_fraction=0.1,
            categorical_as_woe=False
        )

    with pytest.raises(Warning):
        feature_selection_with_iv(
            dfs=df_causal_inference_iv,
            col_target='treat',
            num_features=['age', 'educ'],
            cat_features=None,
            floor_iv=0.3,
            bucket_fraction=0.1,
            categorical_as_woe=False
        )