import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

def prepare_input(text):
  # prepare inputs
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
    input_ids = torch.tensor(tokenizer.encode(PADDING_TEXT + " " + text, add_special_tokens=False)).unsqueeze(0)  
    mask_id = tokenizer.encode("<mask>",add_special_tokens=False)[0]
    target_pos = [(input_ids == mask_id).nonzero(as_tuple=True)[1].item()]
    return input_ids, target_pos

def generate(text):
    # generate model outputs
    input_ids,target_pos = prepare_input(text)
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, target_pos] = 1.0
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
    target_mapping[:,:,target_pos] = 1.0
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    logits = outputs[0]  
    return input_ids,target_pos,logits


def get_score(text,scoreWords):
    input_ids,target_pos,logits = generate(text)
    predictTokens = [tokenizer.encode(word,add_special_tokens=False) for word in words] 
    log_probs = torch.log_softmax(logits,-1).flatten()
    scores = {}
    for predictToken in predictTokens:
        score = torch.sum(log_probs[torch.tensor(predictToken)])
        predictWord = tokenizer.decode(predictToken)
        scores[predictWord] = score.item()
    return scores

def get_topk(text,topk):
    input_ids,target_pos,logits = generate(text)
    log_probs = torch.log_softmax(logits,-1).flatten()
    _,topk_idx = torch.topk(logits,k=topk)
    scores = {}
    for idx in topk_idx.flatten():
        score = log_probs[idx]
        topk_word = tokenizer.decode(idx)
        scores[topk_word] = score.item()
    return scores



