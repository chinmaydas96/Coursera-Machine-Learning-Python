import numpy as np


def getVocabList():

    """reads the fixed vocabulary list in vocab.txt
    and returns a cell array of the words in vocabList.
    """

## Read the fixed vocabulary list
    with open('vocab.txt') as f:

# Store all dictionary words in cell array vocab{}

# For ease of implementation, we use a struct to map the strings => integers
# In practice, you'll want to use some form of hashmap
        vocabList = []
        for line in f:
            idx, w = line.split()
            vocabList.append(w)

    return vocabList
