import json
import torch
from src.language_model import LanguageModel
from src.tokenizer import CharTokenizer




def load_model(config_path):
    
    with open(config_path,"r") as f:
        config = json.load(f)
        
        
        
    with open(config["vocab_path"],"r") as f:
        vocab = json.load(f)
        
        
    model = LanguageModel(d_model=config["d_model"],num_heads=config["num_heads"],num_layers=config["num_layers"],max_len=config["max_len"],vocab_size=config["vocab_size"])
    
    model.load_state_dict(
        torch.load(config["model_path"],map_location="cpu")
    )
    
    
    model.eval()
    
    return model,vocab

def load_tokenizer(config_path):
    tokenizer = CharTokenizer()
    
    with open(config_path,"r") as f:
        config = json.load(f)
    
    tokenizer.load(config["vocab_path"])
    
    return tokenizer
# def load_tokenizer(tokenizer,input):
    
#     input_list = []
#     for s in input:
        
#         id = tokenizer.char2int[s]
        
#         input_list.append(id)
        
        
#     return torch.tensor([[input_list]],dtype=torch.long,device="cpu")
        
    
    
