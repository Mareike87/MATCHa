import pandas


def read_file(file_name, delimiter):
    if delimiter is None:
        delimiter = ','
    return pandas.read_csv(file_name, delimiter=delimiter)


def read_headers(file_name, delimiter):
    if delimiter is None:
        delimiter = ','
    return pandas.read_csv(file_name, delimiter=delimiter).columns.values.tolist()