"""Module Dedicated to Association Metrics

The class implemented in this module is based on the book:
Morettin, P.A. and Bussab, W.O., 2017. Estatística básica. Saraiva Educação SA.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyspark
import pyspark.pandas as ps
from typeguard import typechecked
from typing import List, Union

class Association():
    """This association class implements different types of association metrics, for both categorical and numerical variables.
    
    The class implemented in this module is based on the book:
    Morettin, P.A. and Bussab, W.O., 2017. Estatística básica. Saraiva Educação SA.

    The current implementation is built on top of Koalas, but in the future this will be change to pure pyspark.
    """
    @typechecked
    def C(
        self, 
        df: pyspark.pandas.frame.DataFrame, 
        columns: List[str], 
        dense: bool=True
    ) -> float:
        """Computes the Contingency Coefficient.
        A non-normalized association metric between two categorical variables.

        Args:
            df (pyspark.pandas.frame.DataFrame): A PandasOnSParkDF (sparkDF.to_pandas_on_spark()) contaning the data in dense form or in grouped form.
            columns (List[str]): List of Strings containing the name of the categorical columns which will be used in the coefficient computation.
            dense (bool, optional): If false it is expected that you have executed this df.groupby(columns).size().unstack(level=0).fillna(0) in the df.
                Defaults to True.

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
        for i in ps.DataFrame(total_marginal).iterrows():
            for j in ps.DataFrame(prob_marginal).iterrows():
                valores_esperados.loc[i[0], j[0]] = total_marginal[i[0]]*prob_marginal[j[0]]

        qui_quadrado = ((num_ocorrencias - valores_esperados).pow(2)/valores_esperados).sum().sum()
        C = np.sqrt(qui_quadrado/(qui_quadrado + n))
        return C
    
    @typechecked
    def T(
        self, 
        df: pyspark.pandas.frame.DataFrame, 
        columns: List[str], 
        dense: bool=True
    ) -> float:
        """Computes the T Metrics.
        A normalized metric of assiciation between two categorical variables.

        Args:
            df (pyspark.pandas.frame.DataFrame): A PandasOnSParkDF (sparkDF.to_pandas_on_spark()) contaning the data in dense form or in grouped form.
            columns (List[str]): List of Strings containing the name of the categorical columns which will be used in the coefficient computation.
            dense (bool, optional): If false it is expected that you have executed this df.groupby(columns).size().unstack(level=0).fillna(0) in the df.

        Returns:
            float: The T Metric.
        """
        if (dense):
            num_ocorrencias = df.groupby(columns).size().unstack(level=0).fillna(0)
        else:
            num_ocorrencias = df
        
        r, s = num_ocorrencias.shape
        n = num_ocorrencias.sum().sum()
        prob_marginal = num_ocorrencias.sum(axis=0)/num_ocorrencias.sum().sum()
        total_marginal = num_ocorrencias.sum(axis=1)
        valores_esperados = num_ocorrencias.copy()
        
        for i in ps.DataFrame(total_marginal).iterrows():
            for j in ps.DataFrame(prob_marginal).iterrows():
                valores_esperados.loc[i[0], j[0]] = total_marginal[i[0]]*prob_marginal[j[0]]

        qui_quadrado = ((num_ocorrencias - valores_esperados).pow(2)/valores_esperados).sum().sum()
        T = np.sqrt((qui_quadrado/n)/(np.sqrt((r-1)*(s-1))))
        return T
    
    @typechecked
    def corr(
        self, 
        df: pyspark.pandas.frame.DataFrame, 
        columns: List[str]
    ) -> float:
        """Computes the Correlation Coefficient.

        Standard Correlation Coefficient.

        Args:
            df (databricks.koalas.frame.DataFrame): A PandasOnSParkDF (sparkDF.to_pandas_on_spark()).
            columns (List[str]): List of numeric column names from which the correlation will computed.

        Returns:
            float: Correlation Coefficient.
        """
        X = df[columns[0]]
        Y = df[columns[1]]
      
        n = X.shape[0]
        X_mean = X.mean()
        Y_mean = Y.mean()
        
        num = (X*Y).sum() - n*X_mean*Y_mean
        den = np.sqrt(((X*X).sum() - n*X_mean*X_mean) * ((Y*Y).sum() - n*Y_mean*Y_mean))
        return num/den
    
    @typechecked
    def R2(
        self, 
        df: pyspark.pandas.frame.DataFrame, 
        categorical: str, 
        numerical: str
    ) -> float:
        """Computes the R2 Metric for one numeric column and one categorical column.

        Metric of association between a numeric and a categorical variables.

        Args:
            df (pyspark.pandas.frame.DataFrame): A PandasOnSParkDF (sparkDF.to_pandas_on_spark()).
            categorical (str): Column name of the categorical column.
            numerical (str): column name of the numerical column.

        Returns:
            float: The R2 Metric.
        """
        # X = df[categorical]
        Y = df[numerical]

        data = df.loc[:, [categorical, numerical]]
        Y_var = Y.var()
        Yi_var = data.groupby(categorical).agg({numerical: ['variance', 'count']})

        Y_var__ = (Yi_var[numerical]['variance']*Yi_var[numerical]['count']).sum()/Yi_var[numerical]['count'].sum()
        R2 = 1 - Y_var__/Y_var
        return R2  
    
    @typechecked
    def association_matrix(
        self, 
        df: pyspark.pandas.frame.DataFrame, 
        categorical_features: Union[List[str], None] = None, 
        numerical_features: Union[List[str], None] = None,
        plot_matrix: bool = True,
        return_matrix: bool = False
    ) -> Union[None, pd.core.frame.DataFrame]:
        """Computes from a df, a list of categorical and a list of numerical variables a normalized association matrix.

        Args:
            df (pyspark.pandas.frame.DataFrame): A PandasOnSParkDF (sparkDF.to_pandas_on_spark()). 
            categorical_features (List[str]): List of column names of the categorical features.
            numerical_features (List[str]): List of Column names of the numerical features.
            plot_matrix (bool, optional): If set False it will not plot the matrix. Defaults to True.
            return_matrix (bool, optional): If set to True it will return the correlation matrix as a pandasDF. Defaults to False.

        raises:
            ValueError: if (categorical_features is None) and (numerical_features is None) is True

        Returns:
            Union[None, pd.core.frame.DataFrame]: Either None, if return_matrix is False, or a PandasDF with the correlation coefficients.
        """

        if (categorical_features is None) and (numerical_features is None):
            raise ValueError('Both categorical_features and numerical_features are of type None. At least one must be List[str].')
        
        if categorical_features is None:
            features = numerical_features
        elif numerical_features is None:
            features = categorical_features
        else:
            features = categorical_features + numerical_features
        
        aMatrix = ps.DataFrame(columns=features, index=features, dtype=np.float64)
        
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
        
        if plot_matrix:
            fig, ax = plt.subplots(figsize=(15, 7))
            sns.heatmap(aMatrix.to_pandas().round(2), xticklabels=aMatrix.columns, yticklabels=aMatrix.columns, ax=ax, annot=True)
            fig.show()

        if (return_matrix):
            return aMatrix.to_pandas()
