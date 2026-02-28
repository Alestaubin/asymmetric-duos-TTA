import os
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
from datetime import datetime

from src.utils.load_utils import load_config
from src.utils import metrics
from src.calibration.temperature import jointly_calibrate_temperature, calibrate_temperature
from src.models.inference import get_model_logits_imagenet_c
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event

def save_result_to_csv(result_dict, output_path):
    """Appends a single result row to the CSV. Creates file/header if it doesn't exist."""
    df = pd.DataFrame([result_dict])
    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

def main():
    parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 2: Metrics Calculation')
    parser.add_argument('--config', type=str, default='cfgs/get_metrics.yaml', help='Path to config file')
    cmd_args = parser.parse_args()
    
    tent_cfg = load_config("cfgs/tent.yaml")
    config = load_config(cmd_args.config)
    
    large_name = config['large_model']
    small_name = config['small_model']
    distortions = config['corruption']['distortions']
    severities = config['corruption']['severity']
    
    # Create unique output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./results/metrics"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics_{large_name}_{small_name}_{timestamp}.csv")

    log_event("="*60)
    log_event(f"PHASE 2 START | File: {output_path}")
    log_event(f"Large: {large_name} | Small: {small_name}")
    log_event("="*60)

    # t_large_fixed, t_small_fixed = 0.9101, 1.3456
    # Tl_joint, Ts_joint = 0.6141, 1.7131
    t_large_fixed = calibrate_temperature(
        *get_model_logits_imagenet_c(large_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val")
    )
    t_small_fixed = calibrate_temperature(
        *get_model_logits_imagenet_c(small_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val")
    )
    Tl_joint, Ts_joint = jointly_calibrate_temperature(
        *get_model_logits_imagenet_c(large_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val"),
        *get_model_logits_imagenet_c(small_name, "none", 0, config['val_path'], batch_size=config['batch_size'], num_workers=config['workers'], split="val")
    )
    
    # ---------------------------------------------------------
    # STEP 3: CLEAN IMAGENET TEST SET (BASELINE)
    # ---------------------------------------------------------
    log_event(">>> Processing Clean ImageNet Test Set...")
    zl_clean, labels_clean = get_model_logits_imagenet_c(
        large_name, "none", 0, config['test_path'], 
        batch_size=config['batch_size'], num_workers=config['workers'], split="test"
    )
    zs_clean, _ = get_model_logits_imagenet_c(
        small_name, "none", 0, config['test_path'], 
        batch_size=config['batch_size'], num_workers=config['workers'], split="test"
    )
    
    zt_clean_dict, _ = get_tent_logits_imagenet_c(
        large_name, "none", [0], config['test_path'], tent_cfg, ts=None
    )
    zt_clean = zt_clean_dict[0]

    zt_clean_dict_ts, _ = get_tent_logits_imagenet_c(
        large_name, "none", [0], config['test_path'], tent_cfg, ts="naive"
    )
    zt_clean_ts = zt_clean_dict_ts[0]
    
    variants_clean = {
        "f_large": zl_clean, "f_small": zs_clean, "tent_f_large": zt_clean,
        "f_large_TS": zl_clean / t_large_fixed, "f_small_TS": zs_clean / t_small_fixed,
        "tent_f_large_TS_naive": zt_clean_ts / t_large_fixed,
        "Duo_Joint_TS": (zl_clean / Tl_joint + zs_clean / Ts_joint) / 2
    }

    for name, logits in variants_clean.items():
        acc = (logits.max(1)[1] == labels_clean).float().mean().item()
        nll = F.cross_entropy(logits, labels_clean).item()
        ece = metrics.ece(logits, labels_clean)
        
        res = {
            'large_model': large_name,
            'small_model': small_name,
            'distortion': 'none', 
            'severity': 0, 
            'variant': name, 
            'accuracy': acc, 
            'nll': nll, 
            'ece': ece
        }
        save_result_to_csv(res, output_path)

    # ---------------------------------------------------------
    # STEP 4: IMAGENET-C DISTORTIONS
    # ---------------------------------------------------------
    log_event(">>> Processing ImageNet-C Corruptions...")
    for d_name in distortions:
        log_event(f"Distortion: {d_name}")
        
        tent_logits_dict, _ = get_tent_logits_imagenet_c(
            large_name, d_name, severities, config['data_path'], tent_cfg
        )
        tent_logits_dict_ts, _ = get_tent_logits_imagenet_c(
            large_name, d_name, severities, config['data_path'], tent_cfg, ts="naive"
        )

        for sev in severities:
            zl, labels = get_model_logits_imagenet_c(large_name, d_name, sev, config['data_path'], 
                                                   batch_size=config['batch_size'], num_workers=config['workers'])
            zs, _      = get_model_logits_imagenet_c(small_name, d_name, sev, config['data_path'], 
                                                   batch_size=config['batch_size'], num_workers=config['workers'])
            zt = tent_logits_dict[sev]
            zt_ts = tent_logits_dict_ts[sev]

            variants = {
                "f_large": zl, "f_small": zs, "tent_f_large": zt,
                "f_large_TS": zl / t_large_fixed, "f_small_TS": zs / t_small_fixed,
                "tent_f_large_TS_naive": zt_ts,
                "Duo_Joint_TS": (zl / Tl_joint + zs / Ts_joint) / 2
            }

            for name, logits in variants.items():
                acc = (logits.max(1)[1] == labels).float().mean().item()
                nll = F.cross_entropy(logits, labels).item()
                ece = metrics.ece(logits, labels)
                
                res = {
                    'large_model': large_name,
                    'small_model': small_name,
                    'distortion': d_name, 
                    'severity': sev, 
                    'variant': name, 
                    'accuracy': acc, 
                    'nll': nll, 
                    'ece': ece
                }
                save_result_to_csv(res, output_path)
            
            log_event(f"  Sev {sev} Saved. Duo Acc: {res['accuracy']:.4f}")

    log_event(f"Finished! Final results at {output_path}")

if __name__ == "__main__":
    main()