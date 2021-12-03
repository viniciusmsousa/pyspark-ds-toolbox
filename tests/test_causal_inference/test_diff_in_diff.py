import pyspark_ds_toolbox.causal_inference.diff_in_diff as did


# Diff-in-Diff
def test_did_estimator(df_did_raw):
    # testing output type
    out = did.did_estimator(
        df=df_did_raw,
        id_col='index',
        y='deposits',
        flag_unit='poa',
        flag_time='jul'
    )
    assert type(out) == dict

    # Testing output keys
    answer = [
        'impacto_medio',
        'n_ids_impactados',
        'impacto',
        'pValueInteraction',
        'r2', 'r2adj',
        'df_with_features',
        'linear_model'
    ]
    assert list(out.keys()) == answer

    # testing element types
    # answer_types = [
    #     np.float64,
    #     int,
    #     np.float64,
    #     float, float, float,
    #     pyspark.sql.dataframe.DataFrame,
    #     pyspark.ml.regression.LinearRegressionModel
    # ]
    # for key, type  in zip(answer, answer_types):
    #     print(type(out[key]))
    #     print(type)
    #     assert type(out[key]) == type