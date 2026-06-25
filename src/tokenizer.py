import string
import json
class CharTokenizer:
    def __init__(self):
      
        
        # Single vocab mapping
        self.char2int = {
          
        }
        
        self.int2char = {
         
        }
        

    
    def build_vocab(self, corpus: str):
        clean = corpus
        unique_chars =set(clean)
        
        for char in unique_chars:
            if char not in self.char2int:
                new_id = len(self.char2int)
                self.char2int[char] = new_id
                self.int2char[new_id] = char
                
    def encode(self, sentence: str) -> list:
       
        encoded = []
        
        clean = sentence
        for char in clean:
            # Lookup char, fallback to UNK ID if missing
            char_id = self.char2int[char]
            encoded.append(char_id)
            
      
        return encoded
    
    def decode(self,char_ids:list) ->str:
        
        chars = []
        
        for char_id in char_ids:
            char = self.int2char[char_id]
            
            chars.append(char)
            
            
        return "".join(chars)
    
    
    def save(self,filepath):
        with open(filepath,"w") as f:
            json.dump(self.char2int,f)
            
            
        
    def load(self,filepath):
        with open(filepath,"r") as f:
            self.char2int = json.load(f)
            
            self.int2char = {idx: char for char, idx in self.char2int.items()}
            
        
    
    