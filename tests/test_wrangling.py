import pandas as pd

import pyspark_ds_toolbox.wrangling as wr

def test_pivot_long(dfs_spark_pivot_long_input, dfs_spark_pivot_long_output):
    dfs_result = wr.reshape.pivot_long(
        dfs=dfs_spark_pivot_long_input,
        key_column_name='linha_dre',
        value_column_name='valor',
        key_columns=['id_conta', 'uf', 'anomes', 'safra_ativacao', 'mes_contrato'],
        value_columns=['carteira', 'spread', 'pis_cofins_iss']
    )

    assert dfs_result.schema == dfs_spark_pivot_long_output.schema

def test_count_missing(df_test_count_missing_input, df_test_count_missing_output):
    result = wr.data_quality.count_percent_missing_rows_per_column(sdf=df_test_count_missing_input)

    pd.testing.assert_frame_equal(result, df_test_count_missing_output)