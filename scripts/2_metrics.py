'''
We have the logits for f_large, f_small, and tent_f_large across all distortions and severities.

We now need to:
1. Load the cached logits for each model/distortion combo, and get the predictions for each of the model variants:
    - f_large
    - f_small
    - tent_f_large 
    - f_large + fixed TS
    - f_small + fixed TS
    - tent_f_large + fixed TS
    - Duo + fixed TS (joint calibration)

2. Calculate metrics for each of these variants across all distortions and severities:
    - Accuracy
    - NLL
    - ECE

'''

import os
import pandas as pd
import torch
import torch.nn.functional as F
from src.utils.load_utils import load_config
from src.utils import metrics
from src.calibration.temperature import jointly_calibrate_temperature, calibrate_temperature
from src.models.inference import get_model_logits_imagenetc
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event
import argparse


def main():
    parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 2: Metrics Calculation')
    parser.add_argument('--config', type=str, default='cfgs/save_logits.yaml', help='Path to config file')
    cmd_args = parser.parse_args()
    
    tent_cfg = load_config("cfgs/tent.yaml")
    tent_cfg = DotMap(tent_cfg)

    # 1. Load Configuration
    config = load_config(cmd_args.config)
    large_name = config['large_model']
    small_name = config['small_model']
    distortions = config['corruption']['distortions']
    severities = config['corruption']['severity']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Validation Logits & Calibrate
    log_event("--- Loading Calibration Data ---")
    zl_val, labels_val = get_model_logits_imagenetc(large_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val")
    zs_val, _          = get_model_logits_imagenetc(small_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val")

    # Get Temperatures
    # Scalar (Independent)
    t_large_fixed = calibrate_temperature(zl_val, labels_val, device)
    t_small_fixed = calibrate_temperature(zs_val, labels_val, device)
    # Joint (Duo)
    Tl_joint, Ts_joint = jointly_calibrate_temperature(zl_val, zs_val, labels_val)

    all_results = []

    # 3. Process All Distortions
    log_event("---Calculating Metrics for All Variants ---")
    for d_name in distortions:
        for sev in severities:
            log_event(f"Processing Distortion: {d_name} | Severity: {sev}...", end="\r")
            # Load cached logits
            zl, labels = get_model_logits_imagenetc(large_name, d_name, sev, config['data_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="test")
            zs, _      = get_model_logits_imagenetc(small_name, d_name, sev, config['data_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="test")
            zt, _      = get_tent_logits_imagenet_c(large_name, d_name, sev, config['data_path'], tent_cfg)

            # Define Variants to test
            variants = {
                "f_large": zl,
                "f_small": zs,
                "tent_f_large": zt,
                "f_large_TS": zl / t_large_fixed,
                "f_small_TS": zs / t_small_fixed,
                "tent_f_large_TS": zt / t_large_fixed,
                "Duo_Joint_TS": (zl / Tl_joint + zs / Ts_joint) / 2
            }

            # Calculate Metrics for each variant
            for name, logits in variants.items():
                acc = (logits.max(1)[1] == labels).float().mean().item()
                nll = F.cross_entropy(logits, labels).item()
                ece = metrics.ece(logits, labels)

                all_results.append({
                    'distortion': d_name,
                    'severity': sev,
                    'variant': name,
                    'accuracy': acc,
                    'nll': nll,
                    'ece': ece
                })

            log_event(f"Processed {d_name} Sev {sev}", end="\r")

    # 4. Save and Summarize
    df = pd.DataFrame(all_results)
    output_path = f"./results/phase2_metrics_{large_name}_{small_name}.csv"
    df.to_csv(output_path, index=False)
    
    log_event(f"\nPhase 2 Complete! Results saved to {output_path}")
    
    # Quick Summary Table
    summary = df.groupby('variant')[['accuracy', 'ece']].mean()
    log_event("\nMean Metrics across all ImageNet-C distortions:")
    log_event(summary)

if __name__ == "__main__":
    main()