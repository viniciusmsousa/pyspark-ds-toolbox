# Change Log

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