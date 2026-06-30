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
with open("data/swahili_clean.txt","r") as f:
    raw_text = f.read()
    
    
split_idx = int(len(raw_text) * 0.9)
train_text = raw_text[:split_idx]
val_text = raw_text[split_idx:]
    
    
tokenizer.build_vocab(raw_text)
tokenizer.save("checkpoints/vocab.json")


with open("data/train.txt", "w") as f:
    f.write(train_text)
with open("data/val.txt", "w") as f:
    f.write(val_text)




#dataset = LanguageModelDataset(filepath="data/swahili_clean.txt",tokenizer=tokenizer)


model = LanguageModel(d_model=128,num_heads=4,num_layers=2,max_len=75,vocab_size=len(tokenizer.int2char))

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)

train_dataset = LanguageModelDataset(filepath="data/train.txt", tokenizer=tokenizer)
val_dataset = LanguageModelDataset(filepath="data/val.txt", tokenizer=tokenizer)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, collate_fn=dynamic_pad_collate, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=dynamic_pad_collate, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10


best_val_loss = float('inf')

history = []

patience_counter = 0


early_stop_patience = 5

for epoch in range(epochs):
    training_loss = 0.0
    correct = 0
    total_steps = 0
    total = 0
    
    start_time = time.time()
    for x,y in train_dataloader:
        
        model.train()
        
        
       
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        
        # print(f"The shape of logits is: {logits.shape}")
        # print(f"The shape of y is : {y.shape}")
        
        # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [32*64, 1000]
        flat_out = logits.view(-1, logits.size(-1))

        # Flatten labels to [Batch * Seq_Len] -> [32*64]
        flat_labels = y.view(-1)
        
        # print(f"The shape of flat out is: {flat_out.shape}")
        # print(f"The shape of y is : {flat_labels.shape}")
        
        loss = criterion(flat_out,flat_labels)
        
        loss.backward()
        
        
        optimizer.step()
        
        
        training_loss += loss.item()
        
        
        _,indices = torch.max(logits.data,2)
        
        
        correct_guesses = (indices == y)
        
        
        correct += correct_guesses.sum().item()
        
        total_steps += 1
        total += y.numel()
        
        # print(f"this is the total:{total_steps}")
        #print(f"this is the total len of dataloader: {len(dataloader)}")
    
        
        
    end_time = time.time()
    
    
    total_time = end_time - start_time
    
    average_training_time = total_time/total_steps
    
    average_training_loss = training_loss/total_steps
    
    training_accuracy = 100 * (correct/total)
    
    print(f"this is the total correct: {correct}")
    print(f"this is the total steps: {total_steps}")
    print(f"this is the total expected labels: {total}")
    print(f"this is the total len of training dataloader: {len(train_dataloader)}")
    print(f"this is the total running loss: {training_loss}")
    print(f"this is the average training loss: {average_training_loss}")
    print(f"this is the average training accuracy: {training_accuracy}")
    
    
    model.eval()
    
    
    total_time = 0
    val_loss = 0
    
    running_loss = 0
    correct = 0
    
    total_steps = 0
    total =0
    
    with torch.no_grad():
        for x,y in val_dataloader:
        
          
            
            
        
            x, y = x.to(device), y.to(device)
            
        
            
            logits = model(x)
            
         
            
            # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [32*64, 1000]
            flat_out = logits.view(-1, logits.size(-1))

            # Flatten labels to [Batch * Seq_Len] -> [32*64]
            flat_labels = y.view(-1)
            
         
            
            loss = criterion(flat_out,flat_labels)
            
          
            
           
            
            
            running_loss += loss.item()
            
            
            _,indices = torch.max(logits.data,2)
            
            
            correct_guesses = (indices == y)
            
            
            correct += correct_guesses.sum().item()
            
            total_steps += 1
            total += y.numel()
            
            
            
            # print(f"this is the total of validation:{total_steps}")
            #print(f"this is the total len of dataloader: {len(dataloader)}")
        
            
            
        end_time = time.time()
        
        
        total_time = end_time - start_time
        
        
    average_validation_time = total_time/total_steps
    
    average_validation_loss = running_loss/total_steps
    
    validation_accuracy = 100 * (correct/total)
    
    
    print(f"this is the total correct: {correct}")
    print(f"this is the total steps: {total_steps}")
    print(f"this is the total expected labels: {total}")
    print(f"this is the total len of validation dataloader: {len(val_dataloader)}")
    print(f"this is the total running loss: {training_loss}")
    print(f"this is the average validation loss: {average_validation_loss}")
    print(f"this is the average validation accuracy: {validation_accuracy}")
    print(f"\n")
    
    metrics = {
        
            "Epoch":epoch + 1,
            "train_loss": average_training_loss,
            "training_acc": training_accuracy,
            "training_time":average_training_time,
            "val_loss":average_validation_loss,
            "val_time": average_validation_time,
            "val_accuracy": validation_accuracy,
            
        }
        
        
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {average_training_loss:.4f} | Train Accuracy: {training_accuracy:.2f}%")
    print(f"Epoch {epoch+1}/{epochs} | Val Loss: {average_validation_loss:.4f} | Val Accuracy: {validation_accuracy:.2f}%")

        
    history.append(metrics)
    
    if average_validation_loss < best_val_loss:
        best_val_loss = average_validation_loss
        patience_counter = 0
        torch.save(model.state_dict(),"./checkpoints/best_model.pt")
    
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stoppng triggered at epoch {epoch + 1}")
            break
        
        
with open(f"./experiments/training_data.json","w") as f:
        json.dump(history,f,indent=4)
        
     
    
        
    
    

        
        
        






