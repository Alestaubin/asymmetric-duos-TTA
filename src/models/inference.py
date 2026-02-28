import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import pandas as pd
import numpy as np
import os
from src.models.model_loader import get_model
from src.utils.load_utils import pickle_cache


@pickle_cache("logits_cache")
def get_model_logits_imagenet_c(model_name, distortion, severity, data_path, batch_size=128, num_workers=4, split="test"):
    """
    Extracts or loads cached logits for a specific model/distortion combo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load and move model to device
    net = get_model(model_name, freeze=True)
    net = net.to(device)
    
    # Use DataParallel only if using CUDA and multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.eval()
    
    # 2. Setup Data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    
    if split == "val":
        root_path = data_path
    elif distortion == "none":
        root_path = data_path
    else:
        root_path = os.path.join(data_path, distortion, str(severity))
        
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    
    # pin_memory should be True only for GPU (CUDA or MPS) to speed up transfer
    use_pin_memory = device.type in ['cuda', 'mps']
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )

    all_logits, all_labels = [], []
    
    # 3. Inference Loop
    with torch.no_grad():
        for data, target in loader:
            # Move data to the specified device (CPU/CUDA/MPS)
            data = data.to(device)
            
            output = net(data)
            
            # Always move back to CPU for storage/pickling
            all_logits.append(output.cpu())
            all_labels.append(target.cpu())
            
    return torch.cat(all_logits), torch.cat(all_labels)
