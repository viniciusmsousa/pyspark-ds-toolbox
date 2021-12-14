import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import databricks.koalas as ks
from typeguard import typechecked
import databricks
from typing import List, Union

class Association():
    @typechecked
    def C(self, 
          df:databricks.koalas.frame.DataFrame, 
          columns:List[str], 
          dense:bool=True) -> float:
        """Computes the Contingency Coefficient.

        Args:
            df (pyspark.koalas.frame.DataFrame): Koalas Dataframe contaning the data in dense form or in grouped form.
            columns (List[str]): List of Strings containing the name of the columns which will be used in the coefficient computation.

        Returns:
            float: The Contingency Coefficient value.
        """
        if (dense):
            num_ocorrencias = df.groupby(columns).size().unstack(level=0).fillna(0)
        else:
            num_ocorrencias = df

        n = num_ocorrencias.sum().sum()
        prob_marginal = num_ocorrencias.sum(axis=0)/num_ocorrencias.sum().sum()
        total_marginal = num_ocorrencias.sum(axis=1)
        valores_esperados = num_ocorrencias.copy()
        for i in ks.DataFrame(total_marginal).iterrows():
            for j in ks.DataFrame(prob_marginal).iterrows():
                valores_esperados.loc[i[0], j[0]] = total_marginal[i[0]]*prob_marginal[j[0]]

        qui_quadrado = ((num_ocorrencias - valores_esperados).pow(2)/valores_esperados).sum().sum()
        C = np.sqrt(qui_quadrado/(qui_quadrado + n))
        return C
    
    @typechecked
    def T(self, 
          df:databricks.koalas.frame.DataFrame, 
          columns:List[str]=[], 
          dense:bool=True) -> float:
        if (dense):
            num_ocorrencias = df.groupby(columns).size().unstack(level=0).fillna(0)
        else:
            num_ocorrencias = df
        
        r, s = num_ocorrencias.shape
        n = num_ocorrencias.sum().sum()
        prob_marginal = num_ocorrencias.sum(axis=0)/num_ocorrencias.sum().sum()
        total_marginal = num_ocorrencias.sum(axis=1)
        valores_esperados = num_ocorrencias.copy()
        
        for i in ks.DataFrame(total_marginal).iterrows():
            for j in ks.DataFrame(prob_marginal).iterrows():
                valores_esperados.loc[i[0], j[0]] = total_marginal[i[0]]*prob_marginal[j[0]]

        qui_quadrado = ((num_ocorrencias - valores_esperados).pow(2)/valores_esperados).sum().sum()
        T = np.sqrt((qui_quadrado/n)/(np.sqrt((r-1)*(s-1))))
        return T
    
    @typechecked
    def corr(self, 
             df:databricks.koalas.frame.DataFrame, 
             columns:List[str]) -> float:
        X = df[columns[0]]
        Y = df[columns[1]]
      
        n = X.shape[0]
        X_mean = X.mean()
        Y_mean = Y.mean()
        
        num = (X*Y).sum() - n*X_mean*Y_mean
        den = np.sqrt(((X*X).sum() - n*X_mean*X_mean) * ((Y*Y).sum() - n*Y_mean*Y_mean))
        return num/den
    
    @typechecked
    def R2(self, 
           df:databricks.koalas.frame.DataFrame, 
           categorical:str, 
           numerical:str) -> float:
        X = df[categorical]
        Y = df[numerical]

        data = df.loc[:, [categorical, numerical]]
        Y_var = Y.var()
        Yi_var = data.groupby(categorical).agg({numerical: ['variance', 'count']})

        Y_var__ = (Yi_var[numerical]['variance']*Yi_var[numerical]['count']).sum()/Yi_var[numerical]['count'].sum()
        R2 = 1 - Y_var__/Y_var
        return R2  
    
    @typechecked
    def association_matrix(self, 
                           df:databricks.koalas.frame.DataFrame, 
                           categorical_features:List[str], 
                           numerical_features:List[str],
                           return_matrix:bool = False) -> Union[None, pd.core.frame.DataFrame]:
        features = categorical_features + numerical_features
        
        aMatrix = ks.DataFrame(columns=features, index=features, dtype=np.float64)
        
        for i in aMatrix.iterrows():
            for j in aMatrix.iteritems():
                if ((i[0] in numerical_features) and j[0] in (numerical_features)):
                    aMatrix.loc[i[0], j[0]] = self.corr(df, [i[0], j[0]])
                elif((i[0] in categorical_features) and j[0] in (categorical_features)):
                    aMatrix.loc[i[0], j[0]] = self.T(df, [i[0], j[0]])
                elif((i[0] in categorical_features) and j[0] in (numerical_features)):
                    aMatrix.loc[i[0], j[0]] = self.R2(df, i[0], j[0])
                elif((i[0] in numerical_features) and j[0] in (categorical_features)):
                    aMatrix.loc[i[0], j[0]] = self.R2(df, j[0], i[0])
        
        
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.heatmap(aMatrix.to_pandas().round(2), xticklabels=aMatrix.columns, yticklabels=aMatrix.columns, ax=ax, annot=True)
        fig.show()

        if (return_matrix):
            return aMatrix.to_pandas()
