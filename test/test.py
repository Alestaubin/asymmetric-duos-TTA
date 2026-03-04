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

def run_tent(model, loader, device): 
    
    sev_logits = []
    sev_labels = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        with torch.enable_grad():
            logits = model(data)
        
        sev_logits.append(logits.detach().cpu())
        sev_labels.append(target.cpu())

    sev_logits = torch.cat(sev_logits)
    sev_labels = torch.cat(sev_labels)
    return sev_logits, sev_labels

# def get_tent_model(model_name, lr, beta, steps, episodic, device):
#     model = get_model(model_name)
#     model.train()
#     model = tent.configure_model(model)
#     params, param_names = tent.collect_params(model)

#     optimizer = optim.Adam(params,
#                     lr=lr,
#                     betas=(beta, 0.999),
#                     weight_decay=0.0)

#     tent_model = tent.Tent(model, optimizer,
#                            steps=steps,
#                            episodic=episodic)

#     # print(f"model for adaptation: {model}")
#     log_event(f"Params for adaptation: {param_names}")
#     log_event(f"Number of params: {len(param_names)}")
#     log_event(f"Optimizer for adaptation: {optimizer}")
#     tented_model = tent_model.to(device)
#     return tented_model

def setup_tent(model, lr, beta, steps, episodic):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """

    # 1. Disable all grads first
    model.requires_grad_(False)
    
    # 2. Enable grads for normalization layers (BN for ResNet, LN for ConvNeXt)
    # This replaces tent.configure_model(model) logic to be more inclusive
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            # Force buffers (like running mean/var) to update even in eval mode if needed
            m.track_running_stats = True 
            
    # 3. Collect only those enabled params
    params = []
    param_names = []
    for nm, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            param_names.append(nm)
    
    if not params:
        raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")

    optimizer = optim.Adam(params,
                    lr=lr,
                    betas=(beta, 0.999),
                    weight_decay=0.0)
    
    # 4. Wrap in TENT
    tent_model = tent.Tent(model, optimizer,
                           steps=steps,
                           episodic=episodic)
    
    log_event(f"Params for adaptation: {len(param_names)}")
    return tent_model

def main():

    # 1. Setup
    config = load_config("cfgs/test.yaml")
    data_path = config['data_path']
    models = ["resnet50"]#["resnet50", "convnext_base", "wide_resnet50_2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in models:
        # Use a single, high-impact distortion for a clear signal
        distortion = "gaussian_noise"
        severity = 3 
        
        log_event(f"Model: {model_name} | Target: {distortion} Sev {severity}")
        
        log_event(">>> Running base model without tent...")
        # Get the base model logits for the target distribution (no adaptation)
        preprocess = trn.Compose([
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if distortion == "none" and severity == 0:
            root_path = data_path 
        else:
            root_path = os.path.join(data_path, distortion, str(severity))
        
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Warning: Path {root_path} not found.")
            
        dataset = dset.ImageFolder(root=root_path, transform=preprocess)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, 
            num_workers=8, pin_memory=True
        )
        
        model = get_model(model_name, freeze=False).to(device)

        logits, labels = run_tent(model, loader, device)
        metrics_base = get_metrics_dict(F.softmax(logits, dim=-1), labels)
        print(metrics_base)

        log_event(">>> Running TENT (Standard)...")
        lrs = [0.001, 0.0001, 0.00001]
        batche_sizes = [128]
        for lr in lrs: 
            for batch_size in batche_sizes:
                loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, 
                        num_workers=8, pin_memory=True
                    )
                print(f"\n--- Testing TENT with LR={lr}, Batch Size={batch_size} ---")
                start = time()
                # tented_model = get_tent_model(model_name=model_name,
                #                         lr = lr,
                #                         beta = 0.9,
                #                         steps = 1,
                #                         episodic = True,
                #                         device=device)
                tented_model = setup_tent(get_model(model_name, freeze=False).to(device), 
                                        lr=lr, beta=0.9, steps=1, episodic=True)
                logits_raw, labels = run_tent(tented_model, loader, device) 
                metrics_raw = get_metrics_dict(F.softmax(logits_raw, dim=-1), labels)
                print(metrics_raw)

if __name__ == "__main__":
    main()