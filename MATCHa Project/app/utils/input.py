import pandas as pd

"""Files are read here"""

# Reads a csv-file of the specified name, sets , as delimiter if none is set
# Returns a pandas table with the elements
def read_file(file_name, delimiter):
    if delimiter is None:
        delimiter = ','
    return pd.read_csv(file_name, delimiter=delimiter)


# Reads solely the first line (headers) of a csv-file
# Returns a pandas table containing these
def read_headers(file_name, delimiter):
    if delimiter is None:
        delimiter = ','
    return pd.read_csv(file_name, delimiter=delimiter).columns.values.tolist()

def read_mappings(file_name):
    df = pd.read_csv(file_name, header=None, names=["A", "B"], delimiter=',')
    return set(map(tuple, df.values))