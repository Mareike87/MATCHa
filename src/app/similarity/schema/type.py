import numpy as np

# Put all possible datatypes into one of the categories
def dtype_to_category(dtype, series=None):
    kind = np.dtype(dtype).kind
    if kind in ('i', 'u'):
        return "Integer"
    if kind == 'f':
        return "Decimal"
    if kind == 'b':
        return "Boolean"
    if kind in ('U', 'S'):
        return "String"
    if kind == 'M':
        return "Date"
    if kind == 'm':
        return "Timedelta"
    if kind == 'O':
        # zusätzliche Prüfung nötig
        if series is not None:
            # Prüfe ob Mehrheit Strings sind
            if series.dropna().map(type).eq(str).mean() > 0.8:
                return "String"
        return "Object"
    return "Object"  # Fallback

# Similarity table to compare types
type_compat = {
    ("Integer", "Integer"): 1,
    ("Integer", "Decimal"): 0.8,
    ("Integer", "Boolean"): 0.25,
    ("Integer", "Object"): 0.25,
    ("Integer", "String"): 0.25,
    ("Integer", "Date"): 0.1,
    ("Integer", "Timedelta"): 0.25,
    ("Decimal", "Decimal"): 1,
    ("Decimal", "Boolean"): 0.1,
    ("Decimal", "Object"): 0.25,
    ("Decimal", "String"): 0.25,
    ("Decimal", "Date"): 0.1,
    ("Decimal", "Timedelta"): 0.25,
    ("Boolean", "Boolean"): 1,
    ("Boolean", "Object"): 0.1,
    ("Boolean", "String"): 0.25,
    ("Boolean", "Date"): 0.1,
    ("Boolean", "Timedelta"): 0.1,
    ("Object", "Object"): 1,
    ("Object", "String"): 0.25,
    ("Object", "Date"): 0.25,
    ("Object", "Timedelta"): 0.25,
    ("String", "String"): 1,
    ("String", "Date"): 0.25,
    ("String", "Timedelta"): 0.1,
    ("Date", "Date"): 1,
    ("Date", "Timedelta"): 0.1,
    ("Timedelta", "Timedelta"): 1,
}

for (a, b), val in list(type_compat.items()):
    type_compat[(b, a)] = val

def find_type_similarity(df1, df2):
    m = df1.shape[1]
    n = df2.shape[1]
    sim = np.zeros((m, n))
    types1 = [dtype_to_category(df1[col].dtype, df1[col]) for col in df1.columns]
    types2 = [dtype_to_category(df2[col].dtype, df2[col]) for col in df2.columns]
    for i in range(m):
        for j in range(n):
            sim[i, j] = type_compat.get((types1[i], types2[j]), 0)
    mask = np.ones((m, n))
    return sim, mask

