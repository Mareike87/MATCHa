import numpy as np
import pandas as pd


# Idee: nimm die k (z.B. 50) häufigsten Einträge einer Spalte und vergleiche Sie mit den k häufigsten Einträgen einer anderen Spalte
# Wichtig: es muss eine Normalisierung stattfinden, um den Vergleich zu verbessern (lowercase, maybe weitere String-Optionen?)
# Wichtig: ist die Kardinalität eines Eintrages zu niedrig (z.B. ist in der Spalte jeder Eintrag unique) nutze das Ergebnis nicht → Mask
    # bzw ist das Ergebnis dann einfach ein Stichproben-Wert?

def get_top_k_entries(column, k, isNumber):
    column = pd.Series(column)
    if not isNumber:
        column = (column.astype(str)
                  .str.lower()
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    ######## WARNING: Hier zerschießt irgendwas die SERIES! CHECK VARIABLE TYPE TRANSFORMATIONS
    val_counts = column.value_counts()
    num_unique_entries = val_counts.count()
    top_k = val_counts.head(k)
    if num_unique_entries/column.count() > 0.8 or top_k.count() == 0:
        return top_k, False
    return top_k, True

def comp_top_k(top_k1, top_k2):
    i = top_k1.index
    j = top_k2.index
    intersection = set(i).intersection(set(j))
    union = set(i).union(set(j))
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

def top_k_sim(df1, df2, k):
    m = df1.shape[1]
    n = df2.shape[1]
    sim = np.zeros((m,n))
    mask = np.ones((m,n))

    top_k1 = []
    valid1 = []
    for col in df1.columns:
        top_k_uno, validu = get_top_k_entries(df1[col], k, pd.api.types.is_numeric_dtype(df1[col]))
        top_k1.append(top_k_uno)
        valid1.append(validu)
    top_k2 = []
    valid2 = []
    for col in df2.columns:
        top_k_dos, validd = get_top_k_entries(df2[col], k, pd.api.types.is_numeric_dtype(df2[col]))
        top_k2.append(top_k_dos)
        valid2.append(validd)

    for i in range(m):
        for j in range(n):
            if valid1[i] and valid2[j]:
                print(type(top_k1[i]))
                print(type(top_k2[j]))
                sim[i, j] = comp_top_k(top_k1[i], top_k2[j])
            else: mask[i, j] = 0

    return sim, mask