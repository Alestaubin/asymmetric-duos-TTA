import os
import torch
import pandas as pd
from itertools import product
from src.utils.load_utils import load_config, save_result_to_csv
from src.calibration.pts import get_joint_pts_model, get_joint_pts_logits
from src.calibration.temperature import jointly_calibrate_temperature
from src.utils.metrics import get_metrics_dict
from src.utils.log_utils import log_event
from src.models.inference import get_model_logits_imagenet_c
import torch.nn.functional as F

def main():
    # 1. Setup
    config = load_config('cfgs/get_metrics.yaml')
    large_model = config['large_model']
    small_model = config['small_model']
    val_path = config['val_path']
    
    # 2. Define the Hyperparameter Grid
    # We focus on LR and Hidden Dim as these most impact JPTS stability
    lrs = [1e-3, 5e-4, 1e-4]
    hidden_dims = [128, 256, 512]
    batch_sizes = [128, 256]
    
    output_path = "./results/jpts_tuning_results.csv"
    
    log_event("="*60)
    log_event(f"HYPERPARAMETER TUNING | {large_model} + {small_model}")
    log_event("="*60)

    results = []
    best_loss = float('inf')
    best_acc = 0.0
    best_hyperparams_loss = None
    best_hyperparams_acc = None
    
    zt_large, labels = get_model_logits_imagenet_c(
                                                model_name=large_model, 
                                                distortion="gaussian_noise", 
                                                severity=3,
                                                data_path=config['data_path'], 
                                                batch_size=config['batch_size'], 
                                                num_workers=config['workers'], 
                                                )
    zt_small, _ = get_model_logits_imagenet_c(
                                                model_name=small_model, 
                                                distortion="gaussian_noise", 
                                                severity=3, 
                                                data_path=config['data_path'], 
                                                batch_size=config['batch_size'], 
                                                num_workers=config['workers'], 
                                                )

    # 3. Grid Search Loop
    for lr, bs in product(lrs, hidden_dims, batch_sizes):
        
        # We pass h_dim if your JointPTS class supports it as an argument
        # Assuming you've updated the class to accept hidden_dim
        model, loss = get_joint_pts_model(
            small_model=small_model,
            large_model=large_model,
            data_path=val_path,
            epochs=100, # Max epochs; early stopping will handle the rest
            lr=lr,
            batch_size=bs,
            num_workers=config['workers'],
            patience=7  # Slightly higher patience for tuning
        )
        log_event(f"Completed: LR={lr}, Batch={bs} | Loss: {loss:.6f}")
        
        zd_pts = get_joint_pts_logits(model, zt_large, zt_small)
        metrics = get_metrics_dict(F.softmax(zd_pts, dim=-1), labels)
        log_event(f"Metrics for LR={lr}, Batch={bs}: {metrics}")
        acc = metrics['accuracy']
        if best_acc < acc:  
            best_acc = acc
            best_hyperparams_acc = (lr, bs)
            log_event(f"New Best Accuracy: {best_acc:.4f} with LR={lr}, Batch={bs}")

        if loss < best_loss:
            best_loss = loss
            best_hyperparams_loss = (lr, bs)
            log_event(f"New Best Loss: {best_loss:.6f} with LR={lr}, Batch={bs}")

        res = {
            'lr': lr,
            'batch_size': bs,
            'best_loss': loss,
            'accuracy': acc 
        }
        save_result_to_csv(res, output_path)
        results.append(res)
        

    log_event(f"Tuning complete. Best hyperparameters (loss): {best_hyperparams_loss}")
    log_event(f"Tuning complete. Best hyperparameters (accuracy): {best_hyperparams_acc}")



if __name__ == "__main__":
    main()