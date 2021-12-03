from pytest import fixture
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, LongType, \
    DoubleType, StringType
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F
import pyspark.ml.feature as FF
from pyspark.ml import Pipeline


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

## Wrangling
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
