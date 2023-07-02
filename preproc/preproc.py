import re
import pandas as pd
import numpy as np
import pronouncing
import panphon.distance
import torchtext
from torchtext.vocab import FastText
from torchtext.vocab import GloVe

def fix_contractions(uttr):
    """
    Fix split contractions of the form (e.g., do n't, that 's etc.) in 
    naturalistic speech corpora
    """
    new_tokens = []
    skip_tokens = []
    uttr_tokens = uttr.split()
    for i,uToken in enumerate(uttr_tokens):
        if i < len(uttr_tokens) - 1 and re.search(r"'",uttr_tokens[i+1]):  
            new_token = uToken + uttr_tokens[i+1]
            new_tokens.append(new_token)
            skip_tokens.append(i+1)
        elif i not in skip_tokens:
            new_tokens.append(uToken)
    return " ".join(new_tokens)



def get_subtlexus_freqs(w):
    """
    Get SUBTLEXus word frequencies
    """
    # load frequency table
    freqs = pd.read_csv("SUBTLEXUS.csv")
    freqs["Word"] = freqs["Word"].str.lower()
    words = freqs["Word"].values
    if w in words:
        wf = freqs[freqs["Word"] == w]["FREQcount"].values[0]
        return wf
    else:
        return 0


def phonDist(x,y):
    """
    Computes phonological distance b/w words x and y
    """
    dst = panphon.distance.Distance()
    return dst.feature_edit_distance(x,y)


def phonSim(x,y):
    """
    Computes phonological similarity b/w words x and y
    """
    dst = panphon.distance.Distance()
    phonDist = dst.feature_edit_distance(x,y)
    num_feat1 = dst.feature_edit_distance(x,"")
    num_feat2 = dst.feature_edit_distance(y,"")
    phonSim = 1 - phonDist/max(num_feat1,num_feat2)
    return phonSim



def syllabify(word):
    """
    Returns the syllabic breakdown of a given word
    """
    return pronouncing.phones_for_word(word)


def semDistGloVe(w1,w2):

    """
    Computes the semantic distance between two words
    """
    glove = torchtext.vocab.GloVe(name = '6B', dim = 300)
    x = glove[re.sub(r"['|-]","",w1)].unsqueeze(0)
    y = glove[re.sub(r"['|-]","",w2)].unsqueeze(0)
    semDist = 1-torch.cosine_similarity(x,y)
    return semDist
