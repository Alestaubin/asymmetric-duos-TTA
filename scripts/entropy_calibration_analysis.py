import os
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from dotmap import DotMap

from src.utils.load_utils import load_config, save_result_to_csv
from src.utils.metrics import get_metrics_dict
from src.calibration.temperature import calibrate_temperature
from src.models.inference import get_model_logits_imagenet_c
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event

def get_avg_entropy(logits: torch.Tensor) -> float:
    """Calculates the average Shannon entropy for a batch of logits."""
    probs = F.softmax(logits, dim=-1)
    # 1e-6 epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
    return entropy.mean().item()

def main():
    # 1. Setup Configuration
    config = load_config('cfgs/get_metrics.yaml')
    large_name = config['large_model']
    # Focus on a single representative distortion and severity for depth
    distortion = "gaussian_noise"
    severity = 3 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./results/entropy_analysis_{large_name}_{timestamp}.csv"
    log_event(f"Starting Entropy-Calibration Analysis for {large_name}")

    # 2. Get Source (Val) Data & Global Temperature
    log_event(">>> Processing Source Data (Clean Val)...")
    zl_val, labels_val = get_model_logits_imagenet_c(
        model_name=large_name, distortion="none", severity=0,
        data_path=config['val_path'], batch_size=config['batch_size'], split="val"
    )
    t_global = calibrate_temperature(zl_val, labels_val)
    log_event(f"Optimal Source Temperature: {t_global:.4f}")

    # 3. Get Target (Shifted) Data
    log_event(f">>> Processing Target Data ({distortion} Sev {severity})...")
    zl_tgt, labels_tgt = get_model_logits_imagenet_c(
        model_name=large_name, distortion=distortion, severity=severity,
        data_path=config['data_path'], batch_size=config['batch_size']
    )

    # 4. Get TENT Adapted Data (Standard and with Naive TS)
    log_event(">>> Running TTA Adaptation...")
    # Standard TENT (Uncalibrated start)
    zt_dict_raw, _ = get_tent_logits_imagenet_c(large_name, distortion, severity, config['data_path'], config, ts=None)
    # TENT with Naive TS (Calibrated start)
    zt_dict_ts, _ = get_tent_logits_imagenet_c(large_name, distortion, severity, config['data_path'], config, ts="naive")
    zt_dict_pts, _ = get_tent_logits_imagenet_c(large_name, distortion, severity, config['data_path'], config, ts="pts")

    
    zt_raw = zt_dict_raw[severity]
    zt_ts = zt_dict_ts[severity]
    zt_pts = zt_dict_pts[severity]

    # --- Metrics Calculation ---
    
    # 2.1 Avg Entropy
    analysis_data = []
    
    # Uncalibrated Entropy
    analysis_data.append({
        'ID': '2.1', 'Metric': 'Avg Entropy (Uncalibrated)',
        'Source': get_avg_entropy(zl_val),
        'Target': get_avg_entropy(zl_tgt)
    })
    # Calibrated Entropy
    analysis_data.append({
        'ID': '2.2', 'Metric': 'Avg Entropy (Calibrated)',
        'Source': get_avg_entropy(zl_val / t_global),
        'Target': get_avg_entropy(zl_tgt / t_global)
    })
    analysis_data.append({
        'ID': '2.3', 'Metric': 'Avg Entropy (TENT Raw)',
        'Source': get_avg_entropy(zl_val / t_global),
        'Target': get_avg_entropy(zt_raw)
    })
    analysis_data.append({
        'ID': '2.4', 'Metric': 'Avg Entropy (TENT + Naive TS)',
        'Source': get_avg_entropy(zl_val / t_global),
        'Target': get_avg_entropy(zt_ts)
    })
    analysis_data.append({
        'ID': '2.5', 'Metric': 'Avg Entropy (TENT + PTS)',
        'Source': get_avg_entropy(zl_val / t_global),
        'Target': get_avg_entropy(zt_pts)
    })

    # 2.3 ECE Comparison (Reliability)
    m_base_ts = get_metrics_dict(F.softmax(zl_tgt / t_global, dim=1), labels_tgt)
    m_tent_raw = get_metrics_dict(F.softmax(zt_raw, dim=1), labels_tgt)
    m_tent_ts = get_metrics_dict(F.softmax(zt_ts, dim=1), labels_tgt)
    m_tent_pts = get_metrics_dict(F.softmax(zt_pts, dim=1), labels_tgt)

    log_event("-" * 60)
    log_event("RELIABILITY COMPARISON ON TARGET DATA")
    log_event(f"Source + TS:   Acc {m_base_ts['accuracy']:.4f} | ECE {m_base_ts['ece']:.4f} | Brier {m_base_ts['brier']:.4f}")
    log_event(f"Standard TENT: Acc {m_tent_raw['accuracy']:.4f} | ECE {m_tent_raw['ece']:.4f} | Brier {m_tent_raw['brier']:.4f}")
    log_event(f"TS-TENT:       Acc {m_tent_ts['accuracy']:.4f} | ECE {m_tent_ts['ece']:.4f} | Brier {m_tent_ts['brier']:.4f}")
    log_event(f"PTS-TENT:      Acc {m_tent_pts['accuracy']:.4f} | ECE {m_tent_pts['ece']:.4f} | Brier {m_tent_pts['brier']:.4f}")
    log_event("-" * 60)

    # Save to CSV for plotting
    df = pd.DataFrame(analysis_data)
    # Ratio calculation
    df['Ratio_Delta'] = df['Target'] / df['Source']
    df.to_csv(output_path, index=False)
    log_event(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()