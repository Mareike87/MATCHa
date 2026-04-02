import numpy as np
import re

import Levenshtein

# A basic Levenshtein similarity for two String Lists.
# Returns the similarity matrix
def lev_similarity(df1, df2):
    sim_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            str1 = re.sub(r"\s+", "", df1[i].lower().strip())
            str2 = re.sub(r"\s+", "", df2[j].lower().strip())
            distance = Levenshtein.distance(str1, str2)
            sim_matrix[i][j] = 1 - distance/max(len(str1), len(str2))
    return sim_matrix, np.ones((len(df1), len(df2)))


# Calculates the jaccard similarity for two arrays of strings
# Returns a similarity matrix
def jaccard_sim(df1, df2, token_size=3):
    if token_size < 2 or token_size > 20:
        print("Invalid token size, token size will be set to 3.")
        token_size = 3
    sim_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            sim_matrix[i][j] = jaccard_word(df1[i], df2[j], token_size)
    return sim_matrix, np.ones((len(df1), len(df2)))


# Calculates the jaccard similarity for two strings, using tokens of size token_size
def jaccard_word(str1, str2, token_size=3):
    str1 = re.sub(r"\s+", "", str1.lower().strip())
    str2 = re.sub(r"\s+", "", str2.lower().strip())
    same_tokens = 0
    str1 = " "*(token_size-1) + str1 + " "*(token_size-1)
    str2 = " "*(token_size-1) + str2 + " "*(token_size-1)
    tokens1 = set()
    tokens2 = set()
    for i in range(len(str1)-(token_size-1)):
        tokens1.add(str1[i:i+token_size])
    for i in range(len(str2)-(token_size-1)):
        tokens2.add(str2[i:i+token_size])
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)