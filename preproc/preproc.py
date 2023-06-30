import re

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