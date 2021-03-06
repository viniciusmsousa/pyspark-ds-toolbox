from pytest import fixture
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, LongType, \
    DoubleType, StringType
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F
from pyspark.ml import Pipeline


from pyspark_ds_toolbox.ml.data_prep.features_vector import get_features_vector


@fixture
def spark():
    spark = SparkSession.builder\
                    .appName('Ml-Pipes') \
                    .master('local[1]') \
                    .config('spark.executor.memory', '1G') \
                    .config('spark.driver.memory', '1G') \
                    .config('spark.memory.offHeap.enabled', 'true') \
                    .config('spark.memory.offHeap.size', '1G') \
                    .getOrCreate()
    return spark

# wrangling
@fixture
def dfs_spark_pivot_long_input(spark):
    return spark.createDataFrame(pd.read_csv('tests/data/spark_pivot_long_input.csv'))

@fixture
def dfs_spark_pivot_long_output(spark):
    return spark.createDataFrame(pd.read_csv('tests/data/spark_pivot_long_output.csv'))

@fixture
def df_test_count_missing_input(spark):
    dfs = spark.createDataFrame(
        pd.DataFrame.from_dict({
            'col1': [1, 2, 3, 4],
            'col2': [None, 2, 3, 4],
            'col3': [None, None, 3, 4],
            'col4': [None, None, None, 4]
            }))
    return dfs

@fixture
def df_test_count_missing_output():
    df_out = pd.read_csv('tests/data/df_test_count_missing_output.csv').drop(columns=['Unnamed: 0'])
    df_out.index = pd.Index(['col4', 'col3', 'col2', 'col1'], dtype='object')
    return df_out

# ml.data_prep
@fixture
def schema_get_features_vector():
    schema = StructType([
        StructField('index', LongType(), True),
        StructField('num1', DoubleType(), True),
        StructField('num2', DoubleType(), True),
        StructField('cat1', StringType(), True),
        StructField('cat2', StringType(), True),
        StructField('features', VectorUDT(), True)
    ])
    return schema

@fixture
def df_spark_features_col(spark):
    d = pd.DataFrame({
    'index':[1, 2, 3, 4],
    'num1': [0.1, 0.2, 0.3, 0.4],
    'num2': [0.4, 0.3, 0.2, 0.1],
    'cat1': ['a', 'b', 'a', 'b'],
    'cat2': ['c', 'd', 'c', 'd']
    })
    return spark.createDataFrame(d)

# ml.eval
@fixture
def dfs_decile_analysis_input(spark):
    return spark.createDataFrame(pd.read_csv('tests/data/df_test_binary_classifier_decile_analysis.csv'))

@fixture
def dfs_decile_analysis_output():
    df = pd.read_csv('tests/data/df_test_binary_classifier_decile_analysis_output.csv', sep=';', decimal='.')
    df['percentile'] = df['percentile'].to_numpy('int32')
    return df

@fixture
def dfs_binary_classificator_evaluator(spark):
    df = pd.DataFrame({
        'target': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'predicted': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    })
    return spark.createDataFrame(df)

# causal_inference.ps_matching
@fixture
def df_causal_inference(spark):
    df = pd.read_csv('tests/data/df_causal_inference.csv')
    df.reset_index(level=0, inplace=True)

    df = spark.createDataFrame(df.drop(columns=['data_id']))\
        .withColumn('age2', F.col('age')**2)\
        .withColumn('age3', F.col('age')**3)\
        .withColumn('educ2', F.col('educ')**2)\
        .withColumn('educ_re74', F.col('educ')*F.col('re74'))\
        .withColumn('u74', F.when(F.col('re74')==0, 1).otherwise(0))\
        .withColumn('u75', F.when(F.col('re75')==0, 1).otherwise(0))

    features=['age', 'age2', 'age3', 'educ', 'educ2', 'marr', 'nodegree', 'black', 'hisp', 're74', 're75', 'u74', 'u75', 'educ_re74']
    pipeline = Pipeline(stages = get_features_vector(num_features=features))
    df_assembled = pipeline.fit(df).transform(df)
    return df_assembled

@fixture
def df_ps(spark):
    return spark.createDataFrame(pd.read_csv('tests/data/df_ps.csv'))

@fixture
def df_did_raw(spark):
    dat = pd.read_csv('tests/data/billboard_impact.csv')
    dat.reset_index(level=0, inplace=True)
    df = spark.createDataFrame(dat)
    return df

@fixture
def input_estimate_individual_shapley_values(df_causal_inference):
    train_size=0.8
    train, test = df_causal_inference.randomSplit([train_size, (1-train_size)], seed=12345)
    row_of_interest = df_causal_inference.filter(F.col('index')==3).first()

    return (train, test, row_of_interest, df_causal_inference)

# Stats
@fixture
def ks_iris(spark):
    iris_df = pd.read_csv('tests/data/df_iris.csv')
    iris_sdf = spark.createDataFrame(iris_df)
    iris_ks = iris_sdf.to_pandas_on_spark()

    return iris_ks

# ml.feature_selection.information_value
@fixture
def df_causal_inference_iv(df_causal_inference):
    df = df_causal_inference\
            .withColumn('etnia', F.expr('case when black=1 then "black" when hisp=1 then "hisp" when marr=1 then "marr" else "other" end'))\
            .withColumn('treat', F.col('treat').cast('int'))\
            .withColumn('dumb_cat', F.expr('case when index > 10 then "a" else "b" end'))\
            .select('index', 'age', 'educ', 'etnia','dumb_cat', 'treat')
    return df