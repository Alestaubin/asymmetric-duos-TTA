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
    parser = argparse.ArgumentParser(description='Getting metrics for PTS and TS variants')
    parser.add_argument('--config', type=str, default='cfgs/get_metrics.yaml', help='Path to config file')
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
    output_path = os.path.join(output_dir, f"metrics_{large_name}_{small_name}_{timestamp}.csv")

    log_event("="*60)
    log_event(f"METRICS EXPERIMENT START | File: {output_path}")
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
    log_event(f">>> Calibrating temperature for large model {large_name}")
    t_large_fixed = calibrate_temperature(
        zt_val_large, labels
    )
    log_event(f">>> Calibrating temperature for small model {small_name}")
    t_small_fixed = calibrate_temperature(
        zt_val_small, labels
    )
    log_event(f">>> Calibrating joint temperatures for duo...")
    Tl_joint, Ts_joint = jointly_calibrate_temperature(
        zt_val_large, zt_val_small, labels
    )
    log_event(f"Naive TS -> T_large: {t_large_fixed:.4f} | T_small: {t_small_fixed:.4f}")
    log_event(f"Naive Joint TS -> Tl_joint: {Tl_joint:.4f} | Ts_joint: {Ts_joint:.4f}")

    # ---------------------------------------------------------
    # Get the logits for the clean test set for both models 
    # ---------------------------------------------------------

    log_event(">>> Processing Clean ImageNet Test Set...")
    # TODO: I'm pretty sure that the data is always loaded in the same order for all models, 
    # but should double check this to ensure that the labels are aligned when we compute metrics 
    # across variants. 
    # For making duos, it's assumed e.g. that the i'th logits for small and large corresponds to 
    # the same image.
    zl_clean, yl_clean = get_model_logits_imagenet_c(
                                            model_name=large_name, 
                                            distortion="none", 
                                            severity=0, 
                                            data_path=config['test_path'], 
                                            batch_size=config['batch_size'], 
                                            num_workers=config['workers'], 
                                            split="test"
                                            )
    zs_clean, ys_clean = get_model_logits_imagenet_c(
                                            model_name=small_name, 
                                            distortion="none", 
                                            severity=0, 
                                            data_path=config['test_path'], 
                                            batch_size=config['batch_size'], 
                                            num_workers=config['workers'],
                                            split="test"
                                            )
    zt, yt_clean = get_tent_logits_imagenet_c(
                                            model_name=large_name, 
                                            distortion="none", 
                                            severity=0, 
                                            data_path=config['test_path'], 
                                            cfg=config, 
                                            ts=None
                                            )
    zt_ts, yt_ts_clean = get_tent_logits_imagenet_c(
                                            model_name=large_name, 
                                            distortion="none",
                                            severity=0,
                                            data_path=config['test_path'], 
                                            cfg=config, 
                                            ts="naive"
                                            )
    zt_pts, yt_pts_clean = get_tent_logits_imagenet_c(
                                            model_name=large_name, 
                                            distortion="none", 
                                            severity=0, 
                                            data_path=config['test_path'], 
                                            cfg=config, 
                                            ts="pts"
                                            )

    # ---------------------------------------------------------
    # Pass the logits through the PTS models to get calibrated logits for the clean test set
    # ---------------------------------------------------------

    log_event(">>> Calibrating Clean Test Set Logits with PTS models...")
    zl_clean_pts = get_pts_logits(large_pts_model, zl_clean)
    zs_clean_pts = get_pts_logits(small_pts_model, zs_clean)
    zd_clean_pts = get_joint_pts_logits(joint_pts_model, zl_clean, zs_clean)
    
    variants_clean = {
                "f_large": (zl_clean, yl_clean), 
                "f_small": (zs_clean, ys_clean), 
                "f_large_TS": (zl_clean / t_large_fixed, yl_clean), 
                "f_small_TS": (zs_clean / t_small_fixed, ys_clean),
                "f_large_PTS": (zl_clean_pts, yl_clean), 
                "f_small_PTS": (zs_clean_pts, ys_clean),
                "Duo_Joint_TS": ((zl_clean / Tl_joint + zs_clean / Ts_joint) / 2, yl_clean),
                "Duo_Joint_PTS": (zd_clean_pts, yl_clean),
                "tent_f_large": (zt, yt_clean),
                "tent_f_large_TS_naive": (zt_ts, yt_ts_clean),
                "tent_f_large_TS_pts": (zt_pts, yt_pts_clean),
    }

    for name, results in variants_clean.items():

        logits, labels = results
        metrics_dict = get_metrics_dict(probs=F.softmax(logits, dim=1), labels=labels)
        
        res = {
            'large_model': large_name,
            'small_model': small_name,
            'distortion': 'none', 
            'severity': 0,
            'variant': name, 
            'accuracy': metrics_dict['accuracy'], 
            'nll': metrics_dict['nll'], 
            'ece': metrics_dict['ece'],
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
        for sev in severities:
            zl, yl = get_model_logits_imagenet_c(
                                    model_name=large_name, 
                                    distortion=d_name, 
                                    severity=sev, 
                                    data_path=config['data_path'], 
                                    batch_size=config['batch_size'], 
                                    num_workers=config['workers']
                                    )
            zs, ys      = get_model_logits_imagenet_c(
                                    model_name=small_name, 
                                    distortion=d_name, 
                                    severity=sev, 
                                    data_path=config['data_path'], 
                                    batch_size=config['batch_size'], 
                                    num_workers=config['workers']
                                    )
            zt, yt = get_tent_logits_imagenet_c(
                                    model_name=large_name, 
                                    distortion=d_name, 
                                    severity=sev, 
                                    data_path=config['data_path'], 
                                    cfg=config, 
                                    ts=None
                                    )
            zt_ts, yt_ts = get_tent_logits_imagenet_c(
                                    model_name=large_name, 
                                    distortion=d_name, 
                                    severity=sev, 
                                    data_path=config['data_path'], 
                                    cfg=config, 
                                    ts="naive"
                                    )
            zt_pts, yt_pts = get_tent_logits_imagenet_c(
                                    model_name=large_name, 
                                    distortion=d_name, 
                                    severity=sev, 
                                    data_path=config['data_path'], 
                                    cfg=config, 
                                    ts="pts"
                                    )
            variants = {
                "f_large": (zl, yl), 
                "f_small": (zs, ys), 
                "f_large_TS": (zl / t_large_fixed, yl), 
                "f_small_TS": (zs / t_small_fixed, ys),
                "f_large_PTS": (get_pts_logits(large_pts_model, zl), yl), 
                "f_small_PTS": (get_pts_logits(small_pts_model, zs), ys),
                "Duo_Joint_TS": ((zl / Tl_joint + zs / Ts_joint) / 2, yl),
                "Duo_Joint_PTS": (get_joint_pts_logits(joint_pts_model, zl, zs), yl),
                "tent_f_large": (zt, yt),
                "tent_f_large_TS_naive": (zt_ts, yt_ts),
                "tent_f_large_TS_pts": (zt_pts, yt_pts),
            }

            for name, results in variants.items():
                logits, labels = results
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
                    'f1': metrics_dict['f1'],
                    'brier': metrics_dict['brier']
                }
                save_result_to_csv(res, output_path)
            
            log_event(f"  Sev {sev} Saved. Duo Acc: {res['accuracy']:.4f}")

    log_event(f"Finished! Final results at {output_path}")

if __name__ == "__main__":
    main()