import torch
from torch.nn.utils.rnn import pad_sequence

def dynamic_pad_collate(batch):
    # 1. Rotate the matrix 
    x,y = zip(*batch)
    
    # 2. THE FIX: Cast the raw Python lists into PyTorch Tensors
    x_tensor = [torch.tensor(seq, dtype=torch.long) for seq in x]
    y_tensor = [torch.tensor(seq, dtype=torch.long) for seq in y]
  
    # 3. Pass the valid Tensors into the C++ padding function(if len not same)
    # x_padded = pad_sequence(x_tensor, batch_first=True, padding_value=0)
    # y_padded = pad_sequence(y_tensor, batch_first=True, padding_value=0)
    
    #3. if len is same, then use torch.stack
    x_stacked = torch.stack(x_tensor)
    y_stacked = torch.stack(y_tensor)

    
    return x_stacked,y_stacked