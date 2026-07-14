import torch
import torch.nn as nn
import json
from src.language_model import LanguageModel
from src.tokenizer import CharTokenizer
from src.dataset import LanguageModelDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = CharTokenizer()

tokenizer.load("./checkpoints/gpu/vocab.json")

#dataset = LanguageModelDataset(filepath="data/swahili_clean.txt",tokenizer=tokenizer)

model = LanguageModel(d_model=256,num_heads=8,num_layers=2,max_len=100,vocab_size=len(tokenizer.int2char))

model.load_state_dict(torch.load("./checkpoints/gpu/best_model.pt",map_location=device))

model.eval()


start_id = tokenizer.char2int["S"]

x = torch.tensor([[start_id]],dtype=torch.long,device=device)

out_tensor = model.generate(x,max_new_chars=200)


final_ids = out_tensor.squeeze().cpu().tolist()


output = tokenizer.decode(final_ids)

print(output)





with open("./experiments/gpu/greedy_examples.txt","w",encoding="utf-8") as f:
        f.write(output)
        
        
        
# The temperatures required by the curriculum
temperatures = [0.5, 1.0, 1.5, 2.0]
all_results = []

for t in temperatures:
    print(f"Generating with Temperature = {t}...")
    
    # Re-create a fresh starting tensor for every loop
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    
    # Generate
    out_tensor = model.generate(x, max_new_chars=200, temperature=t)
    final_ids = out_tensor.squeeze().cpu().tolist()
    output = tokenizer.decode(final_ids)
    
    # Format the block
    formatted_output = f"=== T = {t} ===\nPrompt: 'S'\nOutput:\n{output}\n\n"
    all_results.append(formatted_output)

# Save to the specific experiments file
file_path = "./experiments/gpu/temperature_examples.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(all_results)

print(f"\nAll generations saved successfully to {file_path}")





TOP_K = [5,20,50]


all_results = []

for k in TOP_K:
    print(f"Generating with TOP K = {k}...")
    
    # Re-create a fresh starting tensor for every loop
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    
    # Generate
    out_tensor = model.generate(x, max_new_chars=200, temperature=None,k=k,p=None)
    final_ids = out_tensor.squeeze().cpu().tolist()
    output = tokenizer.decode(final_ids)
    
    # Format the block
    formatted_output = f"=== TOP K = {k} ===\nPrompt: 'S'\nOutput:\n{output}\n\n"
    all_results.append(formatted_output)

# Save to the specific experiments file
file_path = "./experiments/gpu/topk_examples.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(all_results)

print(f"\nAll generations saved successfully to {file_path}")


TOP_P = [0.7,0.8,0.9]


all_results = []

for P in TOP_P:
    print(f"Generating with TOP P = {P}...")
    
    # Re-create a fresh starting tensor for every loop
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    
    # Generate
    out_tensor = model.generate(x, max_new_chars=200, temperature=None,k=None,p=P)
    final_ids = out_tensor.squeeze().cpu().tolist()
    output = tokenizer.decode(final_ids)
    
    # Format the block
    formatted_output = f"=== TOP P = {P} ===\nPrompt: 'S'\nOutput:\n{output}\n\n"
    all_results.append(formatted_output)

# Save to the specific experiments file
file_path = "./experiments/gpu/topp_examples.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(all_results)
    
  

with open("./experiments/gpu/config.json","w") as f:
        config = {
            "vocab_size":len(tokenizer.int2char),
            "d_model":256,
            "num_heads":8,
            "num_layers":2,
            "max_len":100,
            "model_path":"./checkpoints/gpu/best_model.pt",
            "vocab_path":"./checkpoints/gpu/vocab.json"
        }
        json.dump(config,f)
            

print(f"\nAll generations saved successfully to {file_path}")








