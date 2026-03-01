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
from src.utils.log_utils import log_event
from src.calibration.pts import get_joint_pts_model, get_pts_logits, get_pts_model, get_joint_pts_logits

def save_result_to_csv(result_dict, output_path):
    """Appends a single result row to the CSV. Creates file/header if it doesn't exist."""
    df = pd.DataFrame([result_dict])
    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

def main():
    parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 2: Metrics Calculation')
    parser.add_argument('--config', type=str, default='cfgs/pts.yaml', help='Path to config file')
    cmd_args = parser.parse_args()
    
    config = load_config(cmd_args.config)
    
    large_name = config['large_model']
    small_name = config['small_model']
    distortions = config['corruption']['distortions']
    severities = config['corruption']['severity']
    
    # Create unique output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./results/metrics"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pts_{large_name}_{small_name}_{timestamp}.csv")

    log_event("="*60)
    log_event(f"PTS START | File: {output_path}")
    log_event(f"Large: {large_name} | Small: {small_name}")
    log_event("="*60)

    zt_val_large, labels = get_model_logits_imagenet_c(
                                                    model_name=large_name, 
                                                    distortion="none", 
                                                    severity=0,
                                                    data_path=config['val_path'], 
                                                    batch_size=config['batch_size'], 
                                                    num_workers=config['workers'], 
                                                    split="val"
                                                    )
    zt_val_small, _ = get_model_logits_imagenet_c(
                                                model_name=small_name, 
                                                distortion="none", 
                                                severity=0, 
                                                data_path=config['val_path'], 
                                                batch_size=config['batch_size'], 
                                                num_workers=config['workers'], 
                                                split="val"
                                                )

    # ---------------------------------------------------------
    # Get the PTS models 
    # ---------------------------------------------------------

    large_pts_model = get_pts_model(model_name=large_name, 
                                    data_path=config['val_path'], 
                                    epochs=config['PTS']['SINGLE']['epochs'], 
                                    lr=config['PTS']['SINGLE']['lr'], 
                                    batch_size=config['PTS']['SINGLE']['batch_size'])
    small_pts_model = get_pts_model(model_name=small_name, 
                                    data_path=config['val_path'], 
                                    epochs=config['PTS']['SINGLE']['epochs'], 
                                    lr=config['PTS']['SINGLE']['lr'], 
                                    batch_size=config['PTS']['SINGLE']['batch_size'])
    joint_pts_model = get_joint_pts_model(small_model=small_name, 
                                        large_model=large_name, 
                                        data_path=config['val_path'], 
                                        epochs=config['PTS']['JOINT']['epochs'], 
                                        lr=config['PTS']['JOINT']['lr'], 
                                        batch_size=config['PTS']['JOINT']['batch_size'])
    
    # ---------------------------------------------------------
    # Get the logits for the clean test set for both models 
    # ---------------------------------------------------------

    log_event(">>> Processing Clean ImageNet Test Set...")
    zl_clean, labels_clean = get_model_logits_imagenet_c(
                                                    model_name=large_name, 
                                                    distortion="none", 
                                                    severity=0, 
                                                    data_path=config['test_path'], 
                                                    batch_size=config['batch_size'], 
                                                    num_workers=config['workers'], 
                                                    split="test"
    )
    zs_clean, _ = get_model_logits_imagenet_c(
        model_name=small_name, 
        distortion="none", 
        severity=0, 
        data_path=config['test_path'], 
        batch_size=config['batch_size'], 
        num_workers=config['workers'],
        split="test"
    )

    # ---------------------------------------------------------
    # Pass the logits through the PTS models to get calibrated logits for the clean test set
    # ---------------------------------------------------------

    log_event(">>> Calibrating Clean Test Set Logits with PTS models...")
    zl_clean_pts = get_pts_logits(large_pts_model, zl_clean)
    zs_clean_pts = get_pts_logits(small_pts_model, zs_clean)
    zd_clean_pts = get_joint_pts_logits(joint_pts_model, zl_clean, zs_clean)


    

    variants_clean = {
        "f_large": zl_clean, "f_small": zs_clean, 
        "f_large_TS": zl_clean_pts, "f_small_TS": zs_clean_pts,
        "Duo_Joint_TS": zd_clean_pts
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
        
        for sev in severities:
            zl, labels = get_model_logits_imagenet_c(model_name=large_name, distortion=d_name, severity=sev, data_path=config['data_path'], batch_size=config['batch_size'], num_workers=config['workers'])
            zs, _      = get_model_logits_imagenet_c(model_name=small_name, distortion=d_name, severity=sev, data_path=config['data_path'], batch_size=config['batch_size'], num_workers=config['workers'])
            variants = {
                "f_large": zl, "f_small": zs, 
                "f_large_TS": get_pts_logits(large_pts_model, zl), "f_small_TS": get_pts_logits(small_pts_model, zs),
                "Duo_Joint_TS": get_joint_pts_logits(joint_pts_model, zl, zs)
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