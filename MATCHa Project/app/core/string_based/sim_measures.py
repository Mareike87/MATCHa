import numpy as np

import Levenshtein

def lev_similarity(df1, df2):
    dist_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            dist_matrix[i][j] = Levenshtein.ratio(df1[i], df2[j])
    return dist_matrix

def jaccard_sim(df1, df2, token_size):
    dist_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            dist_matrix[i][j] = jaccard_word(df1[i], df2[j], token_size)
    return dist_matrix



def jaccard_word(str1, str2, token_size):
    str1 = str1.lower()
    str2 = str2.lower()
    same_tokens = 0
    str1 = " "*(token_size-1) + str1 + " "*(token_size-1)
    str2 = " "*(token_size-1) + str2 + " "*(token_size-1)
    tokens1 = []
    tokens2 = []
    for i in range(len(str1)):
        tokens1.append(str1[i:i+token_size])
    for i in range(len(str2)):
        tokens2.append(str2[i:i+token_size])
    for i in range(len(tokens1)):
        for j in range(len(tokens2)):
            if tokens1[i] == tokens2[j]:
                same_tokens += 1
    return same_tokens / (len(tokens1)+len(tokens2)-same_tokens)
