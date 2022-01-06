"""Evaluation toolbox.

Module dedicated to functionalities related to classification evaluation.
"""

import sys
from typing import List
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
        Exception: Any error that is encontered.

    Returns:
        [Dict]: Dict with the following metrics
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
