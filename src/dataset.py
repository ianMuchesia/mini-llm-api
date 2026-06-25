from torch.utils.data import Dataset



class LanguageModelDataset(Dataset):
    def __init__(self,filepath,tokenizer):
        self.seq_len = 64
        self.tokenizer = tokenizer        
        self.data = []
        sentences = None

        with open(filepath, "r", encoding="utf-8") as file:
            sentences = file.read()
            #self.tokenizer.build_vocab(sentences)
            
            
            
        self.data = self.tokenizer.encode(sentences)
            
        
        
        
        
                
    def __len__(self):
        return len(self.data) - self.seq_len
    
    
    def __getitem__(self, index):

       x = self.data[index:index + self.seq_len]
       y = self.data[index + 1:index + self.seq_len + 1]
       
       
       return x,y
        