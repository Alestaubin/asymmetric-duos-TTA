import argparse
import os
import torch
from time import time

from src.models.model_loader import get_model
from src.utils.load_utils import pickle_cache, get_model_state, reset_model
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.models.inference import get_model_logits_imagenet_c
from src.utils.load_utils import load_config
import logging
from dotmap import DotMap 

logger = logging.getLogger(__name__)

'''
PHASE 1: LOGIT EXTRACTION
This script populates the cache with logits for:
1. f_large (Static)
2. f_small (Static)
3. f_large + TENT (Adaptive)
'''

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
    _ = get_model_logits_imagenet_c(model_name, "none", 0, cfg.val_path, split="val")

    # Extract ImageNet-C Logits
    for d_name in cfg.corruption.distortions:
        for sev in cfg.corruption.severity:
            print(f"  Extracting: {d_name} | Severity {sev}...", end="\r")
            # The @pickle_cache inside this function handles the saving logic
            _ = get_model_logits_imagenet_c(model_name, d_name, sev, cfg.data_path)
print("\nStatic logit extraction complete.")

# ---------------------------------------------------------
# STEP 2: ADAPTIVE LOGIT EXTRACTION (TENT f_large)
# ---------------------------------------------------------
print("\n--- Step 2: Extracting Adaptive Logits (TENT) ---")

tent_cfg = load_config("config/tent.yaml")  
# 1. Initialize the TENT model once
tented_model = get_model(cfg.large_model, freeze = False, tent_enabled=True, cfg=tent_cfg)

for d_name in cfg.corruption.distortions:
    # Before starting a new distortion, reset the model to the source state
    try:
        tented_model.reset()
        logger.info("resetting model")
    except:
        logger.warning("not resetting model")

    for sev in cfg.corruption.severity:
        print(f"Adapting and Evaluating: {d_name} | Severity: {sev}")
        timer_start = time()
        # TENT inference (updates model parameters internally)
        logits, labels = get_tent_logits_imagenet_c(tented_model, d_name, sev, cfg.data_path, batch_size=tent_cfg.TEST.BATCH_SIZE, num_workers=tent_cfg.TEST.WORKERS, split="test")
        print(f"  TTA Logits obtained in {time() - timer_start:.2f}s")

print("\n--- Phase 1 Logit Extraction Complete ---")
print(f"All logits are cached in the 'pickles/logits_cache' directory.")
