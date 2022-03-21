"""Module dedicated to features spark vector tools.
"""

from typing import List, Union
from typeguard import typechecked

import pyspark 
import pyspark.ml.feature as FF

@typechecked
def get_features_vector(
    num_features: Union[List[str], None] = None,
    cat_features: Union[List[str], None] = None,
    output_col = 'features'
) -> List:
    """Assembles a features vector to be used with ML algorithms.

    Args:
        num_features (List[str]): List of columns names of numeric features;
        cat_features (List[str]): List of column names of categorical features (StringIndexer);
        output_col (str): name of the output column;

    Raises:
        TypeError: If num_features AND cat_features are os type None.

    Returns:
        [List]: pyspark indexers, encoders and assemblers like a list;
    """
    if (num_features is None) and (cat_features is None):
        raise TypeError('num_features or cat_features must be a List[str]. Both are none.')
    
    if cat_features is not None:
        indexers = [FF.StringIndexer(inputCol = c, outputCol="{0}_indexed".format(c)) for c in cat_features]
        encoders = [FF.OneHotEncoder(inputCol = indexer.getOutputCol(), outputCol = "{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]
        assemblerCat = FF.VectorAssembler(inputCols = [encoder.getOutputCol() for encoder in encoders], outputCol = "cat")
        stagesCat = indexers + encoders + [assemblerCat]
    
    if num_features is not None:
        assemblerNum = FF.VectorAssembler(inputCols = num_features, outputCol = "num")
        stagesNum = [assemblerNum]

    if (num_features is not None) and (cat_features is not None):
        columns = ['num', 'cat']
        stages = stagesCat + stagesNum
    elif num_features is None:
        columns = ['cat']
        stages = stagesCat
    else:
        columns = ['num']
        stages = stagesNum
    
    stagesList = stages + [FF.VectorAssembler(inputCols = columns, outputCol = output_col)]

    return stagesList
