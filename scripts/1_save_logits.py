import argparse
import os
import torch
from time import time

from src.models.model_loader import get_model
from src.utils.load_utils import pickle_cache, get_model_state, reset_model
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.models.inference import get_model_logits_imagenetc
from src.utils.load_utils import load_config
import logging
from dotmap import DotMap 

'''
PHASE 1: LOGIT EXTRACTION
This script populates the cache with logits for:
1. f_large (Static)
2. f_small (Static)
3. f_large + TENT (Adaptive)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 1: Logit Extraction')
parser.add_argument('--config', type=str, default='cfgs/save_logits.yaml', help='Path to config file')
cmd_args = parser.parse_args()

# Load the YAML configuration
cfg = load_config(cmd_args.config)
cfg = DotMap(cfg)
# ---------------------------------------------------------
# STEP 1: STATIC LOGIT EXTRACTION (f_large, f_small)
# ---------------------------------------------------------
print("\n--- Step 1: Extracting Static Logits (Source & ImageNet-C) ---")

for model_name in [cfg.large_model, cfg.small_model]:
    print(f"\nProcessing Backbone: {model_name}")
    
    # Extract Clean Validation Logits (for calibration later)
    print(f"  Extracting Clean Val Logits...")
    _ = get_model_logits_imagenetc(model_name, "none", 0, cfg.val_path, batch_size=cfg.batch_size, num_workers=cfg.workers, split="val")

    # Extract ImageNet-C Logits
    for d_name in cfg.corruption.distortions:
        for sev in cfg.corruption.severity:
            print(f"  Extracting: {d_name} | Severity {sev}...", end="\r")
            # The @pickle_cache inside this function handles the saving logic
            _ = get_model_logits_imagenetc(model_name, d_name, sev, cfg.data_path, batch_size=cfg.batch_size, num_workers=cfg.workers, split="test")
print("\nStatic logit extraction complete.")

# ---------------------------------------------------------
# STEP 2: ADAPTIVE LOGIT EXTRACTION (TENT f_large)
# ---------------------------------------------------------
print("\n--- Step 2: Extracting Adaptive Logits (TENT) ---")

tent_cfg = load_config("cfgs/tent.yaml")
tent_cfg = DotMap(tent_cfg)

target_severities = cfg.corruption.severity  # e.g., [1, 2, 3, 4, 5]

for d_name in cfg.corruption.distortions:
    print(f"\nProcessing Distortion: {d_name}")
    timer_start = time()
    
    logits_dict, labels_dict = get_tent_logits_imagenet_c(
        model_name=cfg.large_model, 
        distortion_name=d_name, 
        severities=target_severities, 
        data_path=cfg.data_path, 
        tent_cfg=tent_cfg
    )
    
    print(f"Trajectory for {d_name} (Severities {target_severities}) obtained in {time() - timer_start:.2f}s")

print("\n--- Phase 1 Logit Extraction Complete ---")
print(f"All logits are cached in the 'pickles/logits_cache' directory.")
