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
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,batch_first=True,norm_first=True) 
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
    
    def generate(self,x,max_new_chars,temperature=None,k=None,p=None):
        
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
                
                if(temperature is not None):
                      # 1. Apply temperature to the raw logits
                    extracted = extracted / temperature

                    # 2. Convert to perfect percentages (0.0 to 1.0)
                    probs = torch.softmax(extracted, dim=-1)
                    
                    samples = torch.multinomial(probs, num_samples=1)
                    
               
                    
                elif(k is not None ):
                    
                    top_values ,_= torch.topk(extracted,k)
                    
                    threshold = top_values[:,-1]
                    
                    masked_logits = torch.where(extracted < threshold, -float('inf'), extracted)
                    
                    probs = torch.nn.functional.softmax(masked_logits, dim=-1)
                    
                    samples = torch.multinomial(probs, num_samples=1)
                elif(p is not None):
                    #We need to sort the entire 75-character list of percentages to start calculating our running total.
                    probs = torch.softmax(extracted,dim=-1)
                    
                    #we want the biggest, best percentages (like 80%) at the absolute front so the bouncer can add them up first
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    
                    #the bouncer needs to start adding them together one by one to see when the total hits our limit
                    cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
                    
                    
                    #create a variable called mask that checks which items inside cumulative_probs are strictly greater than p
                    sorted_mask = cumulative_probs > p
                    
                    
                    # 4. Shift the mask right to protect the #1 prediction from getting kicked out
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False

                    # 5. Move the True/False values back to their original alphabetical spots
                    mask = torch.zeros_like(extracted, dtype=torch.bool)
                    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

                    # 6. Apply the mask: change garbage scores to negative infinity
                    extracted[mask] = -float('inf')

                    # 7. Final standard dice roll
                    probs = torch.nn.functional.softmax(extracted, dim=-1)
                    samples = torch.multinomial(probs, num_samples=1)
                                        
                    
                    
                    
                    
                    
                    
                    
                else:
                    
                    
                    _,index =  torch.max(extracted.data,dim=1)
                    
                    samples = index.unsqueeze(-1)
                    
                    
                    
                    
                    
              
                
                
                x = torch.cat((x,samples),dim=1)
                
                
        return x
        
        