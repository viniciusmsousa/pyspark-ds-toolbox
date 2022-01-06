"""Propensity Score Matching toolbox.

For an introduction on the subject see: https://mixtape.scunning.com/matching-and-subclassification.html#propensity-score-methods
"""
from typeguard import typechecked

import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, \
    LogisticRegression, GBTClassifier 

from pyspark_ds_toolbox.ml.classification.eval import get_p1, binary_classifier_decile_analysis


@typechecked
def compute_propensity_score(
    df: pyspark.sql.dataframe.DataFrame,
    treat: str,
    y: str,
    id: str,
    featuresCol: str = 'features',
    train_size: float = 0.8
) -> tuple:
    """Computes the propensity score for a given treatment based on the features from df.

    Args:
        df (pyspark.sql.dataframe.DataFrame): Dataframe with features and treatment assignment.
        treat (str): Column name os the treatment indicating column. Must have value 0 or 1.
        y (str): Column name of the outcome of interest.
        id (str): Id of obervations.
        featuresCol (str, optional): Assembled column to be used in a pyspark pipelie. Defaults to 'features'.
        train_size (float, optional): Proportion of the dataset to be used in training. Defaults to 0.8.

    Returns:
        tuple: 1 a sparkDF with the id, propensity score, treat and y columns;
               2 a pandasDF with the evaluation of the models used to compute the propensity score. 
    """
    train, test = df.select(id, featuresCol, treat, y).randomSplit([train_size, (1-train_size)], seed=12345)

    spark_classifiers = {
        'logistic_regression': LogisticRegression(featuresCol=featuresCol, labelCol=treat),
        'decision_tree': DecisionTreeClassifier(featuresCol=featuresCol, labelCol=treat),
        'random_forest': RandomForestClassifier(featuresCol=featuresCol, labelCol=treat),
        'gradient_boosting': GBTClassifier(featuresCol=featuresCol, labelCol=treat)
    }
    df_evaluate = pd.DataFrame()

    for classifier_name, classifier in spark_classifiers.items():
        print(f'{classifier_name}: Starting')
        pipeline = Pipeline(stages = [classifier])
        
        # Fit no Modelo e Predict no test
        print(f'{classifier_name}: Fitting Pipeline')
        fitted_classifier = pipeline.fit(train)
        print(f'{classifier_name}: Making Predictions on test data')
        prediction_on_test = fitted_classifier.transform(test)\
            .withColumn('ps', get_p1(F.col('probability')))

        # Metricas e Avaliacao
        eval_decile = binary_classifier_decile_analysis(dfs=prediction_on_test, col_id='index', col_target=treat, col_probability='ps').toPandas()
        max_ks = eval_decile.iloc[eval_decile['ks'].idxmax(),]
        df_temp = pd.DataFrame({
            'model':[classifier_name],
            'ks_max': [max_ks['ks']],
            'at_decile': [max_ks['percentile']],
            'precision': [max_ks['precision_at_percentile']],
            'recall': [max_ks['cum_eventrate']]
        })
        df_evaluate = df_evaluate.append(df_temp)
        
    final_model = spark_classifiers[df_evaluate.iloc[df_evaluate['ks_max'].idxmax()]['model']].fit(df)
    df_final = final_model.transform(df).withColumn('ps', get_p1(F.col('probability')))
    df_final = df_final.select(id, 'ps', treat, y)
    return (df_final, df_evaluate)

@typechecked
def estimate_causal_effect(
    df_ps: pyspark.sql.dataframe.DataFrame,
    y: str,
    treat: str,
    ps: str
) -> float:
    """Function that estimates the ATE based on propensity scores.

    The implementation is based on chapter 5 Matching and Subclassification
    section 5.3 Approximate Matching of the book Causal Inference: the Mixtape
    (https://mixtape.scunning.com/index.html).

    Args:
        df_ps (pyspark.sql.dataframe.DataFrame): SparkDF with outcome of interest, treatment and propensity scores.
        y (str): Column name of the outcome of interest.
        treat (str): Column name indicating receivement of treatment or not.
        ps (str): Column name with propensity score.

    Returns:
        float: The ATE.
    """
    N = df_ps.count()

    df_effect = df_ps\
        .filter(f'{ps} > 0.1 and {ps} < 0.9')\
        .withColumn('d1', F.expr(f'{treat}/{ps}'))\
        .withColumn('d0', F.expr(f'(1-{treat})/(1-{ps})'))

    N = df_effect.count()
    s1 = df_effect.select(F.sum('d1')).collect()[0][0]
    s0 = df_effect.select(F.sum('d0')).collect()[0][0]

    df_effect = df_effect\
        .withColumn('y1', (F.col(treat)*F.col(y)/F.col(ps)) / (s1/N))\
        .withColumn('y0', ((1-F.col(treat))*F.col(y))/(1-F.col(ps)) / (s0/N))\
        .withColumn('ht', F.expr('y1 -y0'))\
        .drop('y1', 'y0', 'd1', 'd0')
    
    return df_effect.select(F.avg('ht')).collect()[0][0]
