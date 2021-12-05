# Pyspark DS Toolbox

The objective of the package is to provide tools that helps the daily work of data science with spark.

## Package Structure
```
pyspark-ds-toolbox
├─ .git/
├─ .github
│  └─ workflows
│     └─ package-tests.yml
├─ .gitignore
├─ LICENSE.md
├─ README.md
├─ examples
│  └─ ml_eval_estimate_shapley_values.ipynb
├─ poetry.lock
├─ pyproject.toml
├─ docs/
├─ pyspark_ds_toolbox
│  ├─ __init__.py
│  ├─ causal_inference
│  │  ├─ __init__.py
│  │  ├─ diff_in_diff.py
│  │  └─ ps_matching.py
│  ├─ ml
│  │  ├─ __init__.py
│  │  ├─ data_prep.py
│  │  └─ eval.py
│  └─ wrangling.py
├─ requirements.txt
└─ tests
   ├─ __init__.py
   ├─ conftest.py
   ├─ data
   ├─ test_causal_inference
   │  ├─ test_diff_in_diff.py
   │  └─ test_ps_matching.py
   ├─ test_ml
   │  ├─ test_data_prep.py
   │  └─ test_ml_eval.py
   ├─ test_pyspark_ds_toolbox.py
   └─ test_wrangling.py
```