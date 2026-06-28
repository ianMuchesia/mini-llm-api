import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    def __init__(self,d_model,num_heads,num_layers,max_len,vocab_size):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        
        self.emb = nn.Embedding(vocab_size,d_model)
        
        self.pos_emb = nn.Embedding(max_len,d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True,norm_first=True) 
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(d_model,vocab_size)
        
        self.ln_f = nn.LayerNorm(d_model)
        
        
        
    def forward(self,x):
        
        out = self.emb(x)
        
        #positional_encoding
        
        seq_len = x.size(1)
        
        positions_ticket = torch.arange(seq_len,device=x.device)
        
        positions = self.pos_emb(positions_ticket)
        
        out = out  + positions
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        for layer in self.layers:
            
            out = layer(out,src_mask=causal_mask)
            
        out = self.ln_f(out)
        out = self.lm_head(out)
        
        
        return out
    
    def generate(self,x,max_new_chars):
        
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_chars):
                
                seq_len = x.size(1)
                
                if seq_len >= self.max_len:
                    x_cropped = x[:,seq_len-self.max_len:]
                else:
                    x_cropped = x
              
                
                logits = self.forward(x_cropped)
                
                extracted = logits[:,-1,:]
                
                _,index =  torch.max(extracted.data,dim=1)
                
                next_char_id = index.unsqueeze(-1)
                
                x = torch.cat((x,next_char_id),dim=1)
                
                
        return x
        
        