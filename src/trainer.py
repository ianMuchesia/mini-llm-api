import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from torch.utils.data import DataLoader
from src.language_model import LanguageModel
from src.dataset import LanguageModelDataset
from src.tokenizer import CharTokenizer
from src.collate import dynamic_pad_collate



tokenizer = CharTokenizer()


#2. Read the raw text once, 
with open("data/swahili.txt","r") as f:
    raw_text = f.read()
    
    
tokenizer.build_vocab(raw_text)
tokenizer.save("checkpoints/vocab.json")



dataset = LanguageModelDataset(filepath="data/swahili.txt",tokenizer=tokenizer)


model = LanguageModel(d_model=128,num_heads=4,num_layers=2,max_len=75,vocab_size=len(tokenizer.int2char))

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)

dataloader = DataLoader(dataset=dataset,batch_size=32,collate_fn=dynamic_pad_collate,shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 15


best_val_loss = float('inf')

for epoch in range(epochs):
    training_loss = 0.0
    correct = 0
    total = 0
    for x,y in dataloader:
        
        model.train()
       
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        
        print(f"The shape of logits is: {logits.shape}")
        print(f"The shape of y is : {y.shape}")
        
        # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [32*64, 1000]
        flat_out = logits.view(-1, logits.size(-1))

        # Flatten labels to [Batch * Seq_Len] -> [32*64]
        flat_labels = logits.view(-1)
        
        loss = criterion(flat_out,flat_labels)
        
        loss.backward()
        
        
        optimizer.step()
        
        
        training_loss += loss.item()
    
        
        break
    break
        
        
        
        






