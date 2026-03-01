import os
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
from datetime import datetime

from src.utils.load_utils import load_config
from src.utils.metrics import get_metrics_dict
from src.calibration.temperature import jointly_calibrate_temperature, calibrate_temperature
from src.models.inference import get_model_logits_imagenet_c
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event
from src.calibration.pts import get_joint_pts_model, get_pts_logits, get_pts_model, get_joint_pts_logits
from src.utils.load_utils import save_result_to_csv

def main():
    parser = argparse.ArgumentParser(description='Temperature scaling and TENT Experiment')
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
    output_path = os.path.join(output_dir, f"tent_ts_{large_name}_{small_name}_{timestamp}.csv")

    log_event("="*60)
    log_event(f"TENT EXP START | File: {output_path}")
    log_event(f"Large: {large_name} | Small: {small_name}")
    log_event("="*60)

    # ---------------------------------------------------------
    # Obtain validation logits for temperature scaling
    # ---------------------------------------------------------

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
    # Temperature scaling
    # ---------------------------------------------------------

    # PTS Models
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
    # Naive TS temperatures    
    t_large_fixed = calibrate_temperature(
        zt_val_large, labels
    )
    t_small_fixed = calibrate_temperature(
        zt_val_small, labels
    )
    Tl_joint, Ts_joint = jointly_calibrate_temperature(
        zt_val_large, zt_val_small, labels
    )
    log_event(f"Naive TS -> T_large: {t_large_fixed:.4f} | T_small: {t_small_fixed:.4f}")
    log_event(f"Naive Joint TS -> Tl_joint: {Tl_joint:.4f} | Ts_joint: {Ts_joint:.4f}")


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
        "f_large_PTS": zl_clean_pts, "f_small_PTS": zs_clean_pts,
        "Duo_Joint_PTS": zd_clean_pts
    }

    for name, logits in variants_clean.items():

        metrics_dict = get_metrics_dict(probs=F.softmax(logits, dim=1), labels=labels_clean)
        
        res = {
            'large_model': large_name,
            'small_model': small_name,
            'distortion': 'none', 
            'severity': 0,
            'variant': name, 
            'accuracy': metrics_dict['accuracy'], 
            'nll': metrics_dict['nll'], 
            'ece': metrics_dict['ece'],
            'tim_ece': metrics_dict['tim_ece'],
            'f1': metrics_dict['f1'],
            'brier': metrics_dict['brier']
        }
        save_result_to_csv(res, output_path)

    # ---------------------------------------------------------
    # STEP 4: IMAGENET-C DISTORTIONS
    # ---------------------------------------------------------
    log_event(">>> Processing ImageNet-C Corruptions...")
    for d_name in distortions:
        log_event(f"Distortion: {d_name}")
        
        tent_logits_dict, _ = get_tent_logits_imagenet_c(
            large_name, d_name, severities, config['data_path'], tent_cfg, ts=None
        )
        tent_logits_dict_ts, _ = get_tent_logits_imagenet_c(
            large_name, d_name, severities, config['data_path'], tent_cfg, ts="naive"
        )
        tent_logits_dict_pts, _ = get_tent_logits_imagenet_c(
            large_name, d_name, severities, config['data_path'], tent_cfg, ts="pts"
        )

        for sev in severities:
            zl, labels = get_model_logits_imagenet_c(large_name, d_name, sev, config['data_path'], 
                                                   batch_size=config['batch_size'], num_workers=config['workers'])
            zs, _      = get_model_logits_imagenet_c(small_name, d_name, sev, config['data_path'], 
                                                   batch_size=config['batch_size'], num_workers=config['workers'])
            zt = tent_logits_dict[sev]
            zt_ts = tent_logits_dict_ts[sev]
            zt_pts = tent_logits_dict_pts[sev]

            variants = {
                "f_large": zl, "f_small": zs, 
                "f_large_TS": zl / t_large_fixed, "f_small_TS": zs / t_small_fixed,
                "f_large_PTS": get_pts_logits(large_pts_model, zl), "f_small_PTS": get_pts_logits(small_pts_model, zs),
                "Duo_Joint_TS": (zl / Tl_joint + zs / Ts_joint) / 2,
                "Duo_Joint_PTS": get_joint_pts_logits(joint_pts_model, zl, zs),
                "tent_f_large": zt,
                "tent_f_large_TS_naive": zt_ts,
                "tent_f_large_TS_pts": zt_pts,
            }

            for name, logits in variants.items():
                metrics_dict = get_metrics_dict(probs=F.softmax(logits, dim=1), labels=labels)
                
                res = {
                    'large_model': large_name,
                    'small_model': small_name,
                    'distortion': d_name, 
                    'severity': sev, 
                    'variant': name, 
                    'accuracy': metrics_dict['accuracy'], 
                    'nll': metrics_dict['nll'], 
                    'ece': metrics_dict['ece'],
                    'tim_ece': metrics_dict['tim_ece'],
                    'f1': metrics_dict['f1'],
                    'brier': metrics_dict['brier']
                }
                save_result_to_csv(res, output_path)
            
            log_event(f"  Sev {sev} Saved. Duo Acc: {res['accuracy']:.4f}")

    log_event(f"Finished! Final results at {output_path}")

if __name__ == "__main__":
    main()