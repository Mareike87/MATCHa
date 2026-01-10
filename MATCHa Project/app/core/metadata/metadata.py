import pandas as pd
import numpy as np

## NOTE TO SELF: shape[0] = Zeilen, shape[1] = Spalten
def find_equal_types(df1, df2):
    #.dtypes = Series mit index Attributname und value Type
    #.values =
    types1 = df1.dtypes.values[:, None]
    types2 = df2.dtypes.values[None, :]

    return (types1 == types2).astype(int)