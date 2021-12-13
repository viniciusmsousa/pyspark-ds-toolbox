from typing import Union, List
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
        id_col: str,
        target_col: str,
        cat_features: Union[None, List[str]],
        sort_metric: str,
        problem_type: str,
        max_mem_size: str = '3G',
        max_models: int = 10,
        max_runtime_secs: int = 60,
        nfolds: int = 5,
        seed: int = 90
    ):
        # Checking for values errors
        if target_col not in df.columns:
            raise ValueError(f'Column {target_col} not in df.columns.')

        if problem_type not in ['classification', 'regression']:
            raise ValueError(f'problem_type: {problem_type}. Should be in ["classification", "regression"].')


        # Params Values
        self.max_mem_size = max_mem_size
        self.df = df
        self.id_col = id_col
        self.target_col = target_col
        self.cat_features = cat_features
        self.sort_metric = sort_metric
        self.problem_type = problem_type
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed
        
        self.df_id = self.df[[self.id_col]]
        # 1) Starting H2O
        self.start_h2o()
        
        # 2) Getting Features Cols
        self.feature_cols = self.get_feature_cols()
        
        # 3) Spliting into Train and Valid
        self.h2o_df = self.as_h2o_df()

        # 4) Training Model
        self.h2o_automl = self.fit_automl()

        # 5) Shap Values
        self.shap_values = self.extract_shap_values()


    def start_h2o(self):
        h2o.init(max_mem_size=self.max_mem_size)

    def get_feature_cols(self):
        return list(set(list(self.df.columns)).difference([self.target_col, self.id_col]))

    def as_h2o_df(self):

        if self.cat_features is not None:
            for c in self.cat_features:
                h2o_df = pd.concat([self.df, pd.get_dummies(self.df[c], drop_first=True, prefix=c)], axis=1)
        else:
            h2o_df = self.df.copy()

        h2o_df = h2o.H2OFrame(h2o_df.drop(columns=[self.id_col]))
        if self.problem_type == 'classification':
            h2o_df[self.target_col] = h2o_df[self.target_col].asfactor()
        
        return h2o_df

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
            training_frame = self.h2o_df,
        )

        return model
    
    def extract_shap_values(self):
        fi = self.h2o_automl.leader.predict_contributions(test_data=self.h2o_df).as_data_frame()
        _features = list(fi.columns)
        _features.remove('BiasTerm')

        # Proportional weights
        fi['MeanDeviance'] = fi.drop(columns=['BiasTerm']).sum(axis=1)
        fi = pd.concat([fi[_features].div(fi['MeanDeviance'], axis=0), fi[['MeanDeviance', 'BiasTerm']]], axis=1)
        fi.drop(columns=['MeanDeviance', 'BiasTerm'], inplace=True)
        
        # Wide to Long
        fi = pd.concat([self.df_id.reset_index(drop=True), fi.reset_index(drop=True)], axis=1)
        fi = pd.melt(frame=fi, id_vars=[self.id_col], ignore_index=True, value_name='shap_value')
        return fi
