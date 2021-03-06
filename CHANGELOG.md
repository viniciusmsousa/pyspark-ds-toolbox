# Change Log

## 0.4.2

### Fixed
* `pyspark_ds_toolbox.ml.feature_selection.information_value.feature_selection_with_iv()`: `bucket_fraction` argument behavior.

### Changed
* `pyspark_ds_toolbox.ml.feature_selection.information_value.feature_selection_with_iv()`: Return `dict[dfs_iv]` from a spark dataframe to `dict[df_iv]` to a pandas df.

## 0.4.1

### Fixed
* `pyspark_ds_toolbox.ml.feature_selection.information_value.feature_selection_with_iv()`: behavior with `num_features` and `cat_features` arguments.

## 0.4.0

### Added

* Added the `pyspark_ds_toolbox.ml.feature_selection.information_value` module and all its functionalities
    * `feature_selection_with_iv()`
    * `compute_woe_iv()`
    * `WeightOfEvidenceComputer()`

## 0.3.4

### Breaking Changes

* `pyspark_ds_toolbox.ml.data_prep.features_vector.get_features_vector`: Now returns a list with pyspark indexers, encoders and assemblers, to used with pipelines.
* `pyspark_ds_toolbox.ml.classification.baseline_classifiers.py`: Models now are returned as pipelines.

## 0.3.3

### Changed

* `pyspark_ds_toolbox.ml.classification.baseline_binary_classfiers` has a `mlflow_experiment_name` argument.


### Fixed

* `pyspark_ds_toolbox.ml.feature_importance.native_spark`.

## 0.3.2

## Changed

* Fuctionalities from module `pyspark_ds_toolbox.wrangling` was refactored into `pyspark_ds_toolbox.wrangling.reshape.py` and `pyspark_ds_toolbox.wrangling.data_quality.py`;
* Fuctionalities from module `pyspark_ds_toolbox.ml.data_prep` was refactored into `pyspark_ds_toolbox.ml.data_prep.class_weights.py` and `pyspark_ds_toolbox.ml.data_prep.features_vector.py`.

## 0.3.1

### Changed

* Module `pyspark_ds_toolbox.ml.classification.baseline_binary_classfiers` now algo return features scores.

## 0.3.0

### Added 

* Module `pyspark_ds_toolbox.ml.feature_importance` with the functions:
    * `extract_features_score()`

### Changed

* Module `pyspark_ds_toolbox.ml.shap_values` became `pyspark_ds_toolbox.ml.feature_importance.shap_values`


## 0.2.0

### Added

* Module pyspark_ds_toolbox.ml.classification

### changed

* Module pyspark_ds_toolbox.ml.eval became pyspark_ds_toolbox.ml.classification.eval

## 0.1.4

### Changed

* [fix] Class pyspark_ds_toolbox.stats.association.Association now can properly receive only numerical or only categorical features.


## 0.1.3

### Added

* CHANGELOG.md file

### Changed

* pyspark dependency is now >=3.2
* Class pyspark_ds_toolbox.stats.association.Association now uses pyspark.pandas.frame.DataFrame instead of databricks.koalas.frame.DataFrame.
