import pandas as pd
import h2o
from h2o.automl import H2OAutoML


class H2OWrapper:
    """
    H2O AutoML Wrapper
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        sort_metric: str,
        problem_type: str,
        train_size: float,
        max_mem_size: str = '3G',
        max_models: int = 10,
        max_runtime_secs: int = 60,
        nfolds: int = 5,
        seed: int = 90
    ):
        if problem_type not in ['classification', 'regression']:
            raise ValueError(f'problem_type: {problem_type}. Should be in ["classification", "regression"].')

        if (train_size > 0.99) | (train_size < 0.01):
            raise ValueError(f'train_size must be between 0.01 and 0.99')

        # Params Values
        self.max_mem_size = max_mem_size
        self.df = df
        self.target_col = target_col
        self.sort_metric = sort_metric
        self.problem_type = problem_type
        self.train_size = train_size
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed

        # 1) Starting H2O
        self.start_h2o()
        
        # 2) Getting Features Cols
        self.feature_cols = self.get_feature_cols()
        
        # 3) Spliting into Train and Valid
        self.train, self.valid = self.train_valid_split()

        # 4) Training Model
        self.h2o_automl = self.fit_automl()

        # 5) Shap Values
        self.shap_values = self.extract_shap_values()


    def start_h2o(self):
        h2o.init(max_mem_size=self.max_mem_size)

    def get_feature_cols(self):
        return list(self.df.columns[self.df.columns != self.target_col])

    def train_valid_split(self):
        train, valid = h2o.H2OFrame(self.df).split_frame(ratios=[self.train_size])
        
        if self.problem_type == 'classification':
            train[self.target_col] = train[self.target_col].asfactor()
            valid[self.target_col] = valid[self.target_col].asfactor()
        
        return train, valid

    def fit_automl(self):
        ## 1) Training Model
        #https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
        model = H2OAutoML(
            sort_metric = self.sort_metric,
            max_models = self.max_models, 
            max_runtime_secs = self.max_runtime_secs,
            nfolds = self.nfolds,
            seed = self.seed,
            include_algos = ["GBM", "XGBoost", "DRF"] # models that have shap values.
        )
        model.train(
            x = self.feature_cols,
            y = self.target_col, 
            training_frame = self.train,
        )

        return model
    
    def extract_shap_values(self):
        fi = self.h2o_automl.leader.predict_contributions(test_data=self.valid).as_data_frame()
        _features = list(fi.columns)
        _features.remove('BiasTerm')

        fi['MeanDeviance'] = fi.drop(columns=['BiasTerm']).sum(axis=1)
        fi = pd.concat([fi[_features].div(fi['MeanDeviance'], axis=0), fi[['MeanDeviance', 'BiasTerm']]], axis=1)
        return fi

