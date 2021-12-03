import pandas as pd
import numpy as np
import pyspark
from pyspark.sql.types import StructField, StructType, FloatType, LongType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

import pyspark_ds_toolbox.causal_inference.ps_matching as ps


# Propensity Score Matching
def test_compute_propensity_score(df_causal_inference):

    df_ps, df_eval = ps.compute_propensity_score(
        df=df_causal_inference,
        y='re78',
        treat='treat',
        id='index',
        featuresCol='features',
        train_size=0.8
    )

    # Testing
    assert type(df_causal_inference) == type(df_ps)
    assert df_causal_inference.count() == df_ps.count()

    schema = StructType([ \
        StructField("index",LongType(),True), \
        StructField("ps",FloatType(),True), \
        StructField("treat",DoubleType(),True), \
        StructField("re78", DoubleType(), True)
    ])
    assert df_ps.schema == schema

    type(df_eval) == pd.core.frame.DataFrame
    df_eval.shape == (4,5)

def test_estimate_causal_effect(df_ps):
    ate = ps.estimate_causal_effect(
        df_ps=df_ps, y='re78', treat='treat', ps='ps'
    )
    assert type(ate) == float
    assert ate > 2000.0
    assert ate < 2100.0
