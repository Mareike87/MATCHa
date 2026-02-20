import numpy as np
import pandas as pd


# Idee: nimm die k (z.B. 50) häufigsten Einträge einer Spalte und vergleiche Sie mit den k häufigsten Einträgen einer anderen Spalte
# Wichtig: es muss eine Normalisierung stattfinden, um den Vergleich zu verbessern (lowercase, maybe weitere String-Optionen?)
# Wichtig: ist die Kardinalität eines Eintrages zu niedrig (z.B. ist in der Spalte jeder Eintrag unique) nutze das Ergebnis nicht → Mask
    # bzw ist das Ergebnis dann einfach ein Stichproben-Wert?

def get_top_k_entries(column, k, isNumber):
    column = pd.DataFrame(column)
    if not isNumber:
        column = (column.astype(str)
                  .str.lower()
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    top_k = column.value_counts().head(k)
    return top_k

def comp_top_k(top_k1, top_k2):
    i = top_k1.index
    j = top_k2.index
    intersection = set(i).intersection(set(j))
    union = set(i).union(set(j))
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

def top_k_sim(df1, df2, k):


    return