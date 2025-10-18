import os, logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
import tensorflow as tf
tf.random.set_seed(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)

import math
from classes import Cust_FineTune_BertClass

MAX_LENGTH = 202
UNIQUE_CHAR = ['!',	'#', '$', '&', "'", '(', ')', '*', '+', ',', '/', ':', ';', '=', '?', '@', '[', ']', '%',
               '-', '_', '~', '.',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',                
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]', '[SEC]', '[ISEC]']

UNIQUE_CHAR_LEN = len(UNIQUE_CHAR)

vocab =  UNIQUE_CHAR + SPECIAL_TOKENS
token_to_id  = {char: idx for idx, char in enumerate(vocab)}
id_to_token = {idx: char for char, idx in token_to_id.items()}
vocab_size = len(vocab)


def predict_preprocess(url):

    # url = urls.iloc[i]
    # labels = phish.iloc[i]
    url = url.strip() #.lower()
    x = url.partition('//')[2]
    
    is_secure = url.partition('//')[0]
    
    if is_secure == 'https:':
        is_secure = '[SEC]'
    else:
        is_secure = '[ISEC]'
    tokens = ['[CLS]'] + [is_secure] + list(x)
            
    if len(tokens) > MAX_LENGTH:
        tokens = tokens[:MAX_LENGTH-1] + [tokens[-1]]
    
    print("Tokens: ", tokens)
    
    input_ids = [token_to_id.get(token, token_to_id['[UNK]']) for token in tokens]
    
    print("Numerical Representation: ", input_ids)
    
    if len(input_ids) < MAX_LENGTH:
        padding = [token_to_id['[PAD]']] * (MAX_LENGTH - len(input_ids))
        input_ids += padding
    
    attention_mask = [1] * len(tokens) + [0] * (MAX_LENGTH - len(tokens))

    # return (torch.tensor(input_ids), torch.tensor(attention_mask))
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        # "labels": torch.tensor(labels, dtype=torch.long),
    }
        
def predict_url(url):
    
    # input_id, attention_mask = predict_preprocess(url)
    batch = predict_preprocess(url)
    
    input_ids = batch['input_ids'].unsqueeze(0).to(device)
    attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
    
    # print(batch['input_ids'].unsqueeze(0).shape, batch['attention_mask'].unsqueeze(0).shape)
        
    # pred, probability = None, None
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        probability = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    return pred, probability    


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = Cust_FineTune_BertClass(pretrained_path=None, num_labels=2)
    model.load_state_dict(torch.load('/home/g21r1638/test/model test/best_model.pt', weights_only=True))
    model.to(device)
    model.eval()
    
    exit = False
    while exit == False:
        url = input("Enter URL ('exit' or 'e' to quit): ")
        if url.lower() == 'exit' or url.lower() == 'e':
            # exit = True
            print("Exiting...")
            break
        pred, prob = predict_url(url)
        # print(f"URL: {url}")
        print(f"Prediction: {'Phishing' if pred == 1 else 'Benign'}")
        print(f"Confidence: {prob[pred]:.4f}")
        print()
    
    # pred, prob = predict_url(url)
    # print(f"URL: {url}")
    # print(f"Prediction: {'Phishing' if pred == 1 else 'Benign'}")
    # print(f"Confidence: {prob[pred]:.4f}")