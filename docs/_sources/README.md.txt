# Pyspark DS Toolbox

<!-- badges: start -->
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pyspark-ds-toolbox.svg)](https://pypi.org/project/pyspark-ds-toolbox/)
[![CodeFactor](https://www.codefactor.io/repository/github/viniciusmsousa/pyspark-ds-toolbox/badge)](https://www.codefactor.io/repository/github/viniciusmsousa/pyspark-ds-toolbox)
[![Codecov test coverage](https://codecov.io/gh/viniciusmsousa/pyspark-ds-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/viniciusmsousa/pyspark-ds-toolbox?branch=main)
[![Package Tests](https://github.com/viniciusmsousa/pyspark-ds-toolbox/actions/workflows/package-tests.yml/badge.svg)](https://github.com/viniciusmsousa/pyspark-ds-toolbox/actions)
<!-- badges: end -->


The objective of the package is to provide a set of tools that helps the daily work of data science with spark. The documentation can be found [here](https://viniciusmsousa.github.io/pyspark-ds-toolbox/index.html).


## Installation

Directly from PyPi:
```
pip install pyspark-ds-toolbox
```

or from github:
```
pip install git+https://github.com/viniciusmsousa/pyspark-ds-toolbox.git
```

## Organization

The package is currently organized in a structure based on the nature of the task, such as data wrangling, model/prediction evaluation, and so on.

```
pyspark_ds_toolbox         # Main Package
├─ causal_inference           # Sub-package dedicated to Causal Inferece
│  ├─ diff_in_diff.py   
│  └─ ps_matching.py    
├─ ml                         # Sub-package dedicated to ML
│  ├─ data_prep.py      
│  ├─ classification          # Sub-package decidated to classification tasks
│  │  ├─ eval.py
│  │  └─ baseline_classifiers.py 
│  └─ feature_importance 
│     ├─ native_spark.py
│     └─ shap_values.py    
├─ wrangling.py        
└─ stats                      # Sub-package dedicated to basic statistic functionalities
   └─ association.py    
```

