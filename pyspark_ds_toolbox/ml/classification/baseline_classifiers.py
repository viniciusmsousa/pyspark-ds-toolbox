"""A module for baseline classifiers.

"""

import warnings
from typing import Union, List
from typeguard import typechecked
from tqdm import tqdm
import pyspark
from pyspark.sql import functions as F
import pyspark.ml.classification as spark_cl
import mlflow

import pyspark_ds_toolbox.ml.classification.eval as cl_eval
from pyspark_ds_toolbox.ml.data_prep.features_vector import get_features_vector 
from pyspark_ds_toolbox.ml.data_prep.class_weights import binary_classifier_weights
from pyspark_ds_toolbox.ml.feature_importance.native_spark import extract_features_score

@typechecked
def baseline_binary_classfiers(
    dfs: pyspark.sql.dataframe.DataFrame,
    dfs_test: pyspark.sql.dataframe.DataFrame,
    id_col: str,
    target_col: str,
    num_features: Union[List[str], None] = None,
    cat_features: Union[List[str], None] = None,
    weight_on_target: bool = False,
    log_mlflow_run: bool = False,
    mlflow_experiment_name: Union[None, str] = None,
    artifact_stage_path: Union[None, str] = None
) -> dict:
    """Function that fit models that could be used as baseline model.

    This function will:
     1) Add a features vecto to dfs and dfs_test (see pyspark_ds_toolbox.ml.classification.data_prep.get_features_vector());
     2) Fit the following models LogisticRegression, DecisionTreeClassifier, RandomForestClassifier and GBTClassifier, without any tunning;
     3) Extracts from the trained models the features score (see pyspark_ds_toolbox.ml.feature_importance.native_spark.extract_features_score);
     4) Use the fitted models to predict on the dfs_test;
     5) Compute evaluation metrics on the test data (see pyspark_ds_toolbox.ml.classification.eval module)

    Args:
        dfs (pyspark.sql.dataframe.DataFrame): A training DataFrameSpark.
        dfs_test (pyspark.sql.dataframe.DataFrame): A test DataFrameSpark. Metrics reported are computed with the prediction values from this data.
        id_col (str): Column name of the id. Used to compute the confusion matrix.
        target_col (str): Target to be predicted. Must be of values 1 and 0.
        num_features (Union[List[str], None], optional): List of the columns names of numerical features. Defaults to None.
        cat_features (Union[List[str], None], optional): List of the columns names of the categorical features. Defaults to None.
        weight_on_target (bool, optional): If True will add a class weight based on target_col (see pyspark_ds_toolbox.ml.data_prep.binary_classifier_weights). Defaults to False.
        log_mlflow_run (bool, optional): If True will log params, metrics, confusion matrix, decile table and model in a MLFlow run for each fit. Defaults to False.
        mlflow_experiment_name (Union[None, str], optional): Name of the experiment where the runs should be looged. Defaults to None.
        artifact_stage_path (Union[None, str], optional): Path to write confusion matrix and decile table before logging into mlflow. Defaults to None.

    Raises:
        ValueError: if dfs.schema != dfs_test.schema is True
        ValueError: len(set([id_col, target_col]).difference(set(dfs.columns))) != 0 is True
     
    Returns:
        [dict]: A dict for each algorithm (keys are LogisticRegression, DecisionTreeClassifier, RandomForestClassifier and GBTClassifier). Each element is a dictionary with the keys
            - model: The spark trained model;
            - feature_score: The feature importance of the model (see pyspark.ml.feature_importance.spark_native.extract_features_score); 
            - metrics: dict with confusion_matrix and f1, auc, accuracy, precision, recall and max_ks;
            - decile_table: Table with a decile analysis on the predicted probabilities.
    """
    if dfs.schema != dfs_test.schema:
        raise ValueError('Condition dfs.schema != dfs_test.schema yield True. dfs and dfs_test must same schema.')
    
    if len(set([id_col, target_col]).difference(set(dfs.columns))) != 0:
        raise ValueError('id_col and target_col must be column names in dfs and dfs_test.')
         
    if (log_mlflow_run is True) and (artifact_stage_path is None):
        warnings.warn('log_mlflow_run is True and artifact_stage_path is None. This means that artifacts (confusion matrix and decile table) will not be logged to mlflow.')

    if (mlflow_experiment_name is not None) and (log_mlflow_run is False):
        warnings.warn('mlflow_experiment_name is not None and log_mlflow_run is False. This means that runs will not be logged into the experiment.')

    if (log_mlflow_run is True) and (mlflow_experiment_name is not None):
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)


    print('Computing Features Vector')
    dfs = get_features_vector(
        df=dfs,
        num_features=num_features,
        cat_features=cat_features
    )
    dfs_test = get_features_vector(
        df=dfs_test,
        num_features=num_features,
        cat_features=cat_features
    )
    
    if weight_on_target:
        print('Computing Class Weights')
        dfs = binary_classifier_weights(dfs=dfs, col_target=target_col)
        dfs_test = binary_classifier_weights(dfs=dfs_test, col_target=target_col)
        weigth_col = f'weight_{target_col}'

        print('Instanciating Classifiers')
        lr = spark_cl.LogisticRegression(labelCol=target_col, featuresCol='features', weightCol=weigth_col).fit(dfs)
        dt = spark_cl.DecisionTreeClassifier(labelCol=target_col, featuresCol='features', weightCol=weigth_col).fit(dfs)
        rf = spark_cl.RandomForestClassifier(labelCol=target_col, featuresCol='features', weightCol=weigth_col).fit(dfs)
        gbt = spark_cl.GBTClassifier(labelCol=target_col, featuresCol='features', weightCol=weigth_col).fit(dfs)
    else:
        print('Instanciating Classifiers')
        lr = spark_cl.LogisticRegression(labelCol=target_col, featuresCol='features').fit(dfs)
        dt = spark_cl.DecisionTreeClassifier(labelCol=target_col, featuresCol='features').fit(dfs)
        rf = spark_cl.RandomForestClassifier(labelCol=target_col, featuresCol='features').fit(dfs)
        gbt = spark_cl.GBTClassifier(labelCol=target_col, featuresCol='features').fit(dfs)    


    print('Predicting on Test Data and Evaluating')
    out_dict = dict()
    names = [
        'LogisticRegression', 'DecisionTreeClassifier',
        'RandomForestClassifier', 'GBTClassifier'
    ]
    models = [lr, dt, rf, gbt]
    for name, model in tqdm(zip(names, models), total=len(names)):

        df_fi = extract_features_score(model=model, dfs=dfs)

        prediction = model.transform(dfs_test)
        metrics = cl_eval.binary_classificator_evaluator(
            dfs_prediction=prediction,
            col_target=target_col,
            col_prediction='prediction',
            print_metrics=False
        )

        decile_metrics = cl_eval.binary_classifier_decile_analysis(
            dfs=prediction.withColumn('p1', cl_eval.get_p1(F.col('probability'))),
            col_id=id_col,
            col_target=target_col,
            col_probability='p1'
        )
        decile_metrics = round(decile_metrics.toPandas(), 4)


        out_dict[name] = {
            'model': model,
            'feature_score': df_fi,
            'metrics': metrics,
            'decile_metrics': decile_metrics
        }

        if log_mlflow_run:
            with mlflow.start_run(run_name=f'{name}'):
                # Log Params
                mlflow.log_param('target', target_col)
                mlflow.log_param('weight_on_target', str(weight_on_target))
                paramMap = {str(k).partition('__')[2]: v for (k, v) in model.extractParamMap().items()}
                mlflow.log_params(paramMap)

                # Log Metrics
                mlflow.log_metric("f1", metrics["f1"])
                mlflow.log_metric("auc", metrics["auroc"])
                mlflow.log_metric('accuracy', metrics["accuracy"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("max_ks", float(decile_metrics[['ks']].max()))

                # Log Artefacts and Model
                if artifact_stage_path is not None:
                    metrics['confusion_matrix'].to_csv(f'{artifact_stage_path}confusion_matrix.csv', index=False, sep='|')
                    mlflow.log_artifact(f'{artifact_stage_path}confusion_matrix.csv')

                    decile_metrics.to_csv(f'{artifact_stage_path}decile_metrics.csv', index=False, sep='|')
                    mlflow.log_artifact(f'{artifact_stage_path}decile_metrics.csv')

                    df_fi.to_csv(f'{artifact_stage_path}feature_score.csv', index=False, sep='|')
                    mlflow.log_artifact(f'{artifact_stage_path}feature_score.csv')



                mlflow.spark.log_model(model, "model")

    return out_dict
