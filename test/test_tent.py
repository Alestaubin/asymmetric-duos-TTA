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
    
    # log_event(">>> Running base model without tent...")
    # # Get the base model logits for the target distribution (no adaptation)
    # logits_base, labels = get_model_logits_imagenet_c(
    #     model_name=model_name, distortion=distortion, severity=severity, 
    #     data_path=config['data_path']
    # )
    # metrics_base = get_metrics_dict(F.softmax(logits_base, dim=-1), labels)
    # print(metrics_base)

    log_event(">>> Running TENT (Standard)...")
    start = time()
    # We use a dummy ts value or force a cache skip if necessary to ensure fresh results
    for ts in [None, "naive", "pts"]:
        zt, yt_clean = get_tent_logits_imagenet_c(
                                                model_name=model_name, 
                                                distortion=distortion, 
                                                severity=severity, 
                                                data_path=config['data_path'], 
                                                cfg=config, 
                                                ts=ts
                                                )
        metrics_raw = get_metrics_dict(F.softmax(zt, dim=-1), yt_clean)
        print(f"TS={ts}: {metrics_raw}")

if __name__ == "__main__":
    main()