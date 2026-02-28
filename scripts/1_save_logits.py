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
from src.calibration.temperature import TemperatureWrapper

'''
PHASE 1: LOGIT EXTRACTION
This script populates the cache with logits for
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 1: Logit Extraction')
parser.add_argument('--config', type=str, default='cfgs/save_logits.yaml', help='Path to config file')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with fewer samples')
parser.add_argument('--model', type=str, default='resnet18', help='the model to use')
parser.add_argument('--tent', action='store_true', default=False, help='Enable tent adaptation for the specified model')
parser.add_argument('--ts', type=str, default=None, help='Type of temperature scaling to apply (if --tent is enabled). Options: "pts", "naive", or None (no scaling).')
cmd_args = parser.parse_args()

# Load the YAML configuration
cfg = load_config(cmd_args.config)

if not cmd_args.tent:
    print("\n--- Extracting Static Logits (Source & ImageNet-C) ---")
    model_name = cmd_args.model

    print(f"\nProcessing Backbone: {model_name}")
    
    # Extract Clean Validation Logits (for calibration later)
    print(f"  Extracting Clean Val Logits...")
    _ = get_model_logits_imagenet_c(model_name, "none", 0, cfg.val_path, batch_size=cfg.batch_size, num_workers=cfg.workers, split="val")

    # Extract ImageNet-C Logits
    for d_name in cfg.corruption.distortions:
        for sev in cfg.corruption.severity:
            print(f"  Extracting: {d_name} | Severity {sev}...", end="\r")
            # The @pickle_cache inside this function handles the saving logic
            _ = get_model_logits_imagenet_c(model_name, d_name, sev, cfg.data_path, batch_size=cfg.batch_size, num_workers=cfg.workers, split="test")
    print("\nStatic logit extraction complete.")
elif cmd_args.tent:
    print("\n---  Extracting Adaptive Logits (TENT) ---")
    print(f"\nProcessing Backbone: {cmd_args.model}")

    tent_cfg = load_config("cfgs/tent.yaml")

    target_severities = cfg.corruption.severity  # e.g., [1, 2, 3, 4, 5]

    for d_name in cfg.corruption.distortions:
        print(f"\nProcessing Distortion: {d_name}")
        timer_start = time()
        
        logits_dict, labels_dict = get_tent_logits_imagenet_c(
            model_name=cmd_args.model, 
            distortion_name=d_name, 
            severities=target_severities, 
            data_path=cfg.data_path, 
            tent_cfg=tent_cfg,
            ts=cmd_args.ts
        )
        
        print(f"Trajectory for {d_name} (Severities {target_severities}) obtained in {time() - timer_start:.2f}s")
