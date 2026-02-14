import numpy as np

import Levenshtein

"""String similarity measures are implemented here."""

# A basic Levenshtein similarity for two String Lists.
# Returns the similarity matrix
def lev_similarity(df1, df2):
    sim_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            sim_matrix[i][j] = Levenshtein.ratio(df1[i], df2[j])
    return sim_matrix, np.full((len(df1), len(df2)), 1)


# Calculates the jaccard similarity for two arrays of strings
# Returns a similarity matrix
# TODO: add exception for invalid token sizes
def jaccard_sim(df1, df2, token_size):
    sim_matrix = np.zeros((len(df1), len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            sim_matrix[i][j] = jaccard_word(df1[i], df2[j], token_size)
    return sim_matrix, np.full((len(df1), len(df2)), 1)


# Calculates the jaccard similarity for two strings, using tokens of size token_size
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

# TODO: finish whatever is going on here
def n_grams(text, n):
    # Transform string to lowercase and add spaces ahead of and behind string
    text = " "*(n-1) + text.lower() + " "*(n-1)
    return {text[i:i+n] for i in range(len(text)-n+1)}