import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import pandas as pd
import numpy as np

from src.models.model_loader import get_model
from src.utils.load_utils import pickle_cache

BATCH=128
WORKERS=4

@pickle_cache("logits_cache")
def get_model_logits_imagenet_c(model_name, distortion, severity, data_path, split="test"):
    """
    Extracts or loads cached logits for a specific model/distortion combo.
    Note: We pass parameters to the decorator via the function arguments.
    """
    # Load model locally inside function to keep the cache clean
    net = get_model(model_name, freeze=True)
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
    net.eval()
    
    # Setup data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    
    if split == "val":
        root_path = data_path # Assumes path to clean val images
    else:
        root_path = os.path.join(data_path, distortion, str(severity))
        
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, 
                                         num_workers=WORKERS, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            output = net(data.cuda())
            all_logits.append(output.cpu())
            all_labels.append(target.cpu())
            
    return torch.cat(all_logits), torch.cat(all_labels)

