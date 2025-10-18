#finetuning bert class 
import torch
from transformers import BertConfig, BertModel
from safetensors import safe_open

MAX_LENGTH = 202
UNIQUE_CHAR = ['!',	'#', '$', '&', "'", '(', ')', '*', '+', ',', '/', ':', ';', '=', '?', '@', '[', ']', '%',
               '-', '_', '~', '.',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',                               
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SPECIAL_TOKENS = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]', '[SEC]', '[ISEC]']
UNIQUE_CHAR_LEN = len(UNIQUE_CHAR)
VOCAB =  UNIQUE_CHAR + SPECIAL_TOKENS
VOCAB_SIZE = len(VOCAB)

class Cust_FineTune_BertClass(torch.nn.Module):
    def __init__(self, pretrained_path, num_labels):
        super(Cust_FineTune_BertClass, self).__init__()
        
        config = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=512,                   
            num_hidden_layers=12,              
            num_attention_heads=8,            
            max_position_embeddings=MAX_LENGTH,
            type_vocab_size=2,  
            hidden_dropout_prob=0.1, #0.2, 0.3
            attention_probs_dropout_prob=0.1
        )
        
        self.bert = BertModel(config)
                
        if not pretrained_path == None and pretrained_path.endswith('.safetensors'):
            pretrained_weights = {}
            with safe_open(pretrained_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    pretrained_weights[key] = f.get_tensor(key)        

            # bert_weights = {k: v for k, v in pretrained_weights.items() 
            #             if not k.startswith('cls.')}
            
            bert_weights = {}
            for k, v in pretrained_weights.items():
                if k.startswith("bert."):
                    new_k = k[len("bert."):] 
                else:
                    new_k = k
                if not new_k.startswith("cls."):
                    bert_weights[new_k] = v

            self.bert.load_state_dict(bert_weights, strict=False)
            
            missing, unexpected = self.bert.load_state_dict(bert_weights, strict=False)
            print("Missing:", missing)
            print("Unexpected:", unexpected)
            
        # pretrained_weights = {}
        # with safe_open(pretrained_path, framework="pt", device="cpu") as f:
        #     for key in f.keys():
        #         pretrained_weights[key] = f.get_tensor(key)        

        # bert_weights = {k: v for k, v in pretrained_weights.items() 
        #             if not k.startswith('cls.')}

        # self.bert.load_state_dict(bert_weights, strict=False)            

        # self.l1 = BertModel.from_pretrained("pretrain.pt")
        
        self.pre_classifier = torch.nn.Linear(512, 512)
        self.dropout = torch.nn.Dropout(0.5) #0.1
        self.classifier = torch.nn.Linear(512, num_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask): #, token_type_ids
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            # token_type_ids=token_type_ids
            )
        
        hidden_state = outputs.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output