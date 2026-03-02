import torch
import torch.nn.functional as F
import numpy as np
from time import time
import dependencies.tent.tent as tent
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as trn
import torchvision.datasets as dset
import os
import torch.nn as nn

from src.utils.metrics import get_metrics_dict
from src.utils.load_utils import load_config
from src.models.model_loader import get_model
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event
from src.models.inference import get_model_logits_imagenet_c
from src.calibration.pts import PTSWrapper, get_pts_model

from src.tta.tent_utils import setup_tent

def get_entropy(logits: torch.Tensor) -> float:
    """Calculates the average entropy of a batch of logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
    return entropy.mean().item()

def main():
    # 1. Setup
    config = load_config("cfgs/get_metrics.yaml")
    
    model_name = config['large_model']
    # Use a single, high-impact distortion for a clear signal
    distortion = "gaussian_noise"
    severity = 3 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_event("="*60)
    log_event(f"INVESTIGATION: TENT vs TENT+TS divergence")
    log_event(f"Model: {model_name} | Target: {distortion} Sev {severity}")
    log_event("="*60)

    model = get_model(model_name="convnext_base", freeze=False)
    
    cfg = config["TENT"]
    tent_model = setup_tent(model, cfg)
    tent_model = tent_model.to(device)

    # ---------------------------------------------------------
    # TEST 1: TENT WITHOUT TS
    # ---------------------------------------------------------
    log_event(">>> Running TENT (Standard)...")
    start = time()

    preprocess = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    root_path = os.path.join(config['data_path'], distortion, str(severity))
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, 
            num_workers=8, pin_memory=True
        )
    sev_logits = []
    sev_labels = []

    # TENT adapts by looking at batches sequentially
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        with torch.enable_grad():
            logits = tent_model(data)
        
        sev_logits.append(logits.detach().cpu())
        sev_labels.append(target.cpu())

    logits_raw = torch.cat(sev_logits)
    labels = torch.cat(sev_labels)
    acc_raw = (logits_raw.max(1)[1] == labels).float().mean().item()
    ent_raw = get_entropy(logits_raw)
    metrics_raw = get_metrics_dict(F.softmax(logits_raw, dim=-1), labels)
    print(metrics_raw)
    log_event(f"Standard TENT -> Acc: {acc_raw:.4f} | Avg Entropy: {ent_raw:.4f} | Time: {time()-start:.2f}s")

if __name__ == "__main__":
    main()