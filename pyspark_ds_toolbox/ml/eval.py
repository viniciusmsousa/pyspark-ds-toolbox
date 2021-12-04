"""Machine Learning (with Spark) Evaluation toolbox.

Module dedicated to functionalities related to ML evaluation.
"""

import sys
from typeguard import typechecked
import pandas as pd

import pyspark 
from pyspark.sql import functions as F, types as T, Window
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT


get_p1 = F.udf(lambda value: value[1].item(), T.FloatType())

@typechecked
def binary_classificator_evaluator(
    dfs_prediction: pyspark.sql.dataframe.DataFrame,
    col_target: str,
    col_prediction: str,
    print_metrics: bool = False
) -> dict:

    """Computes the Matrics of a Binary Classifier from a Prediction Output table from spark.

    Args:
        dfs_prediction (pyspark.sql.dataframe.DataFrame): Output Prediction table from spark binarry classifier.
        col_target (str): Column name with the target (ground truth)
        print_metrics (bool, optional): Wether to print or not the metrics in the console. Defaults to False.

    Raises:
        Exception: Any error that is encontered

    Returns:
        Dict: Dictionery with the following metrics:
            - confusion_matrix
            - accuracy
            - precision
            - recall
            - f1 score
            - aucaoc
    """
    try:
        # Confusion Matrix
        confusion_matrix = dfs_prediction.groupBy(col_target, col_prediction).count() 
        TN = dfs_prediction.filter(f'{col_prediction} = 0 AND {col_target} = 0').count()
        TP = dfs_prediction.filter(f'{col_prediction} = 1 AND {col_target} = 1').count()
        FN = dfs_prediction.filter(f'{col_prediction} = 0 AND {col_target} = 1').count()
        FP = dfs_prediction.filter(f'{col_prediction} = 1 AND {col_target} = 0').count()

        # Computing Metrics from Confusion Matrix
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0 

        evaluator = BinaryClassificationEvaluator(labelCol=col_target, rawPredictionCol=col_prediction, metricName='areaUnderROC')
        aucroc = evaluator.evaluate(dfs_prediction)
        evaluator = BinaryClassificationEvaluator(labelCol=col_target, rawPredictionCol=col_prediction, metricName='areaUnderPR')
        aucpr = evaluator.evaluate(dfs_prediction)

        # Results Dict
        out_dict = {
            'confusion_matrix': confusion_matrix.toPandas(),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'aucroc': aucroc,
            'aucpr': aucpr
        }
        
        if print_metrics == True:
            print(f'accuracy:  {round(out_dict["accuracy"], 4)}')
            print(f'precision: {round(out_dict["precision"], 4)}')
            print(f'recall:    {round(out_dict["recall"], 4)}')
            print(f'f1:        {round(out_dict["f1"], 4)}')
            print(f'auroc:     {round(out_dict["auroc"], 4)}')
        
        return out_dict
    except Exception as e:
        raise Exception(e)

@typechecked
def binary_classifier_decile_analysis(
    dfs: pyspark.sql.dataframe.DataFrame,
    col_id: str,
    col_target: str,
    col_probability: str
) -> pyspark.sql.dataframe.DataFrame:
    """Computes a Precision, Recall and KS decile analysis from a probability prediction model.
    col_target column MUST have values of only 0 and 1.

    Args:
        dfs (pyspark.sql.dataframe.DataFrame): SparkDF with probabilities predictions.
        col_id (str): Column name with id value, to count the values for each decile.
        col_target (str): Column name with the ground truth. Must be from a binary classifier with values 1 and 0.
        col_probability (str): Column name with the probability estimated from the model.

    Raises:
        ValueError: If unique values from col_target column are not 0 and 1.

    Returns:
        pyspark.sql.dataframe.DataFrame: SparkDF with the columns:
            - percentile, min_prob, max_prob, count_id, events, non_events, cum_events,
            cum_non_events, precision_at_percentile, recall_at_percentile, event_rate,
            nonevent_rate, cum_eventrate, cum_noneventrate, ks.
    """
    unique_rows = dfs.select(col_target).distinct().collect()
    unique_rows_values = [v[col_target] for v in unique_rows]
    unique_rows_values.sort()
    
    if unique_rows_values != [0, 1]:
        raise ValueError(f'Unique values from {col_target} column are not [0, 1].')

    # 1 Adding non events column to dataset
    dfs = dfs.withColumn('target0',1 - dfs[col_target])
    
    # 2 Adding the decile to the dataset
    dfs = dfs.withColumn("prob_qcut", F.ntile(10).over(Window.partitionBy().orderBy(dfs[col_probability])))
    
    # 3 Aggragating to compute the counts and min/max probabilities
    decile_table = dfs\
            .groupBy('prob_qcut')\
            .agg(
                F.count(col_id),
                F.avg(col_probability),
                F.min(col_probability),
                F.max(col_probability),
                F.sum(col_target),
                F.sum('target0')
            )\
            .orderBy(F.col(f'min({col_probability})').desc())\
            .withColumnRenamed(f'count({col_id})', 'count_id')\
            .withColumnRenamed(f'avg({col_probability})', 'avg_prob')\
            .withColumnRenamed(f'min({col_probability})', 'min_prob')\
            .withColumnRenamed(f'max({col_probability})', 'max_prob')\
            .withColumnRenamed(f'sum({col_target})', 'events')\
            .withColumnRenamed('sum(target0)','non_events')
    
    # 4 Computing the events and non-events rates
    count_event = dfs.filter(f'{col_target} == 1').count()
    count_nonevent = dfs.filter(f'{col_target} == 0').count()
    decile_table = decile_table.withColumn('event_rate', decile_table.events/count_event)
    decile_table = decile_table.withColumn('nonevent_rate', decile_table.non_events/count_nonevent)
    
    # 5 Cumulating the events/non-evenys values and rates
    win = Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)
    decile_table = decile_table.withColumn('cum_events', F.sum(decile_table.events).over(win))
    decile_table = decile_table.withColumn('cum_non_events', F.sum(decile_table.non_events).over(win))
    decile_table = decile_table.withColumn('cum_eventrate', F.sum(decile_table.event_rate).over(win)) # recall
    decile_table = decile_table.withColumn('cum_noneventrate', F.sum(decile_table.nonevent_rate).over(win))
    
    # 6 Compute the precision at percentiles
    decile_table = decile_table.withColumn('precision_at_percentile', F.col('cum_events')/(F.col('cum_events')+F.col('cum_non_events')))

    # 7 Computing the KS Metric
    decile_table = decile_table.withColumn('ks', F.expr('cum_eventrate - cum_noneventrate'))
    
    # 8 Adding decile number
    decile_table = decile_table.withColumn('percentile', F.row_number().over(Window.orderBy(F.lit(1))))

    # 9 Selecting columns of interest
    decile_table = decile_table.select(
        'percentile', 'min_prob', 'max_prob', 'avg_prob', 'count_id', 'non_events', 'events', 
        'cum_non_events', 'cum_events', 'nonevent_rate', 'event_rate', 'cum_noneventrate',
        'cum_eventrate',  'precision_at_percentile', 'ks'
    )
    return decile_table

def estimate_individual_shapley_values(
        spark,
        df,
        id_column,
        model,
        row_of_interest,
        feature_names,
        column_to_examine,
        features_col='features',
        print_shap_values=False
):
    """
    # Based on the algorithm described here:
    # https://christophm.github.io/interpretable-ml-book/shapley.html#estimating-the-shapley-value
    # And on Baskerville's implementation for IForest/ AnomalyModel here:
    # https://github.com/equalitie/baskerville/blob/develop/src/baskerville/util/model_interpretation/helpers.py#L235
    """

    # 1) Creating empty sdf to host the shap values.
    schema = T.StructType([
        T.StructField(id_column, T.IntegerType(), True),
        T.StructField('feature', T.StringType(), True),
        T.StructField('shap', T.FloatType(), True)
    ])
    results = spark.createDataFrame(spark.sparkContext.emptyRDD(),schema)

    features_perm_col = 'features_permutations'
    marginal_contribution_filter = F.avg('marginal_contribution').alias('shap_value')
    
    # 2) Broadcast the row of interest and ordered feature names
    ROW_OF_INTEREST_BROADCAST = spark.sparkContext.broadcast(row_of_interest)
    ORDERED_FEATURE_NAMES = spark.sparkContext.broadcast(feature_names)

    # 3) Persist before continuing with calculations
    if not df.is_cached:
        df = df.persist()

    # 4) Get permutations
    # Creates a column for the ordered features and then shuffles it.
    # The result is a dataframe with a column `output_col` that contains:
    # [feat2, feat4, feat3, feat1],
    # [feat3, feat4, feat2, feat1],
    # [feat1, feat2, feat4, feat3],
    # ...
    features_df = df.withColumn(
        'features_permutations',
        F.shuffle(
            F.array(*[F.lit(f) for f in feature_names])
        )
    )

    # 5) Set up the udf - x-j and x+j need to be calculated for every row
    def calculate_x(
            feature_j, z_features, curr_feature_perm
    ):
        """
        The instance  x+j is the instance of interest,
        but all values in the order before feature j are
        replaced by feature values from the sample z
        The instance  x−j is the same as  x+j, but in addition
        has feature j replaced by the value for feature j from the sample z
        """
        x_interest = ROW_OF_INTEREST_BROADCAST.value
        ordered_features = ORDERED_FEATURE_NAMES.value
        x_minus_j = list(z_features).copy()
        x_plus_j = list(z_features).copy()
        f_i = curr_feature_perm.index(feature_j)
        after_j = False
        for f in curr_feature_perm[f_i:]:
            # replace z feature values with x of interest feature values
            # iterate features in current permutation until one before j
            # x-j = [z1, z2, ... zj-1, xj, xj+1, ..., xN]
            # we already have zs because we go row by row with the udf,
            # so replace z_features with x of interest
            f_index = ordered_features.index(f)
            new_value = x_interest[f_index]
            x_plus_j[f_index] = new_value
            if after_j:
                x_minus_j[f_index] = new_value
            after_j = True

        # minus must be first because of lag
        return Vectors.dense(x_minus_j), Vectors.dense(x_plus_j)

    udf_calculate_x = F.udf(calculate_x, T.ArrayType(VectorUDT()))

    # persist before processing
    features_df = features_df.persist()

    for f in feature_names:
        # x column contains x-j and x+j in this order.
        # Because lag is calculated this way:
        # F.col('anomalyScore') - (F.col('anomalyScore') one row before)
        # x-j needs to be first in `x` column array so we should have:
        # id1, [x-j row i,  x+j row i]
        # ...
        # that with explode becomes:
        # id1, x-j row i
        # id1, x+j row i
        # ...
        # to give us (x+j - x-j) when we calculate marginal contribution
        # Note that with explode, x-j and x+j for the same row have the same id
        # This gives us the opportunity to use lag with
        # a window partitioned by id
        x_df = features_df.withColumn('x', udf_calculate_x(
            F.lit(f), features_col, features_perm_col
        )).persist()

        # Calculating SHAP values for f
        x_df = x_df.selectExpr(
            id_column, f'explode(x) as {features_col}'
        ).cache()
        x_df = model.transform(x_df).withColumn('probability', get_p1(F.col('probability')))

        # marginal contribution is calculated using a window and a lag of 1.
        # the window is partitioned by id because x+j and x-j for the same row
        # will have the same id
        x_df = x_df.withColumn(
            'marginal_contribution',
            F.col(column_to_examine) - F.lag(F.col(column_to_examine), 1).over(Window.partitionBy(id_column).orderBy(id_column))
        )
        # calculate the average
        x_df = x_df.filter(x_df.marginal_contribution.isNotNull())
        
        feat_shap_value = pd.DataFrame.from_dict({
            id_column: [row_of_interest[id_column]],
            'feature': [f],
            'shap_value': [x_df.select(marginal_contribution_filter).first().shap_value]
        })
        feat_shap_value = spark.createDataFrame(feat_shap_value)
        if print_shap_values:
            print(f'Marginal Contribution for feature: {f} = {x_df.select(marginal_contribution_filter).first().shap_value}')
        
        results = results.union(feat_shap_value)
        
    return (results, x_df)