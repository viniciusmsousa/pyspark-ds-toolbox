import pytest
import pandas as pd

from pyspark_ds_toolbox.stats.association import Association

def test_association(ks_iris):
    A = Association()
    categorical_features = ['target']
    numerical_features = ['sepal length (cm)', 
                        'sepal width (cm)', 
                        'petal length (cm)',
                        'petal width (cm)']

    df = A.association_matrix(
        df=ks_iris, 
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        plot_matrix=False,
        return_matrix=True
    )

    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (5, 5)

    with pytest.raises(ValueError):
        df = A.association_matrix(
            df=ks_iris, 
            categorical_features=None,
            numerical_features=None,
            plot_matrix=True,
            return_matrix=True
        )

def test_assiciation_c(ks_iris):
    c = Association().C(df=ks_iris,columns=['target', 'target'])

    assert 0.816496580927726 == c
