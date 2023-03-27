import re
import torch
import numpy as np
import pandas as pd
from transformers import XLNetTokenizer, XLNetLMHeadModel, XLNetModel


# XLNet
tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased")

def XLNet_LM(text):
  tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
  model.to("cuda:0")
  outputs = model(tokens.unsqueeze(0).to("cuda:0"))
  logits = outputs.logits
  return logits


def XLNet_maskedModel(text,bidirectional=False):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0).to("cuda:0")
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    if bidirectional == False:
      mask_pos = -1
    else:
      mask_idx = tokenizer.encode("<mask>",add_special_tokens=False)[0]
      mask_pos = (input_ids == mask_idx).nonzero()[0][-1].item()
    perm_mask[:, :, mask_pos] = 1.0 
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float).to("cuda:0")
    target_mapping[0, 0, mask_pos] = 1.0
    model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    logits = outputs.logits
    return logits


 def getTokenScore(logits,word):
 	token = tokenizer.encode(word, add_special_tokens=False)
 	return torch.sum(torch.log(torch.softmax(logits[-1,-1],dim=0)[token]))


def compute_alignments(text,tokens):
  orig_tokens = text.split()
  token_list = [tokenizer.decode(token) for token in tokens]
  alignments, _ = tokenizations.get_alignments(orig_tokens,token_list)
  return alignments