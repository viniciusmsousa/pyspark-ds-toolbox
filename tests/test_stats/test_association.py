import pandas as pd

from pyspark_ds_toolbox.stats.association import Association

def test_association(ks_iris):
    A = Association()
    categorical_features = ['target']
    numerical_features = ['sepal length (cm)', 
                        'sepal width (cm)', 
                        'petal length (cm)',
                        'petal width (cm)']

    df = A.association_matrix(ks_iris, 
                              categorical_features, 
                              numerical_features,
                              return_matrix=True)

    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (5, 5)