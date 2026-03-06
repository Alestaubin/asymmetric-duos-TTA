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
from src.utils.plot_utils import plot_epoch_losses, get_time_str
from src.calibration.pts_2 import JointPTS_calibrator

def main():
    # 1. Setup
    config = load_config('cfgs/get_metrics.yaml')
    large_model = config['large_model']
    small_model = config['small_model']
    lr = config['PTS']['JOINT']['lr']
    epochs = config['PTS']['JOINT']['epochs']
    bs = config['PTS']['JOINT']['batch_size']
    num_workers = config['PTS']['JOINT']['num_workers']
    top_k_logits = config['PTS']['JOINT']['top_k_logits']
    weight_decay = config['PTS']['JOINT']['weight_decay']
    patience = config['PTS']['JOINT']['patience']
    log_event(f"Running experiment with large_model={large_model}, small_model={small_model}, lr={lr}, top_k_logits={top_k_logits}, weight_decay={weight_decay}, patience={patience}")
    zt_large, labels = get_model_logits_imagenet_c(
                                                model_name=large_model, 
                                                distortion="gaussian_noise", 
                                                severity=3,
                                                data_path=config['data_path'], 
                                                batch_size=config['batch_size'], 
                                                num_workers=config['workers'], 
                                                )
    zt_large_clean, labels_clean = get_model_logits_imagenet_c(
                                            model_name=large_model, 
                                            distortion="none",
                                            severity=0,
                                            data_path=config['val_path'], 
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
    zt_small_clean, _ = get_model_logits_imagenet_c(
                                            model_name=small_model, 
                                            distortion="none",
                                            severity=0,
                                            data_path=config['val_path'], 
                                            batch_size=config['batch_size'], 
                                            num_workers=config['workers'], 
                                            )
    Tl_joint, Ts_joint = jointly_calibrate_temperature(
        zt_large_clean, zt_small_clean, labels_clean
    )

    clean_duo_logits = (zt_large_clean / Tl_joint + zt_small_clean / Ts_joint) / 2
    clean_duo_probs = F.softmax(clean_duo_logits, dim=1).numpy()
    clean_metrics = get_metrics_dict(clean_duo_probs, labels_clean)
    log_event(f"Metrics on clean data for Joint Temperature Scaling: {clean_metrics}")

    shifted_duo_logits = (zt_large / Tl_joint + zt_small / Ts_joint) / 2
    shifted_duo_probs = F.softmax(shifted_duo_logits, dim=1).numpy()
    shifted_metrics = get_metrics_dict(shifted_duo_probs, labels)
    log_event(f"Metrics on OOD data for Joint Temperature Scaling: {shifted_metrics}")

    for loss_fn in ["mse", "nll"]:
        calibrator = JointPTS_calibrator(epochs=epochs,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        batch_size=bs,
                                        nlayers=2,
                                        n_nodes=256,
                                        length_logits=zt_small.shape[1],
                                        top_k_logits=top_k_logits,
                                        loss_fn=loss_fn)

        calibrator.tune(logits_s=zt_small_clean, logits_l=zt_large_clean, labels=labels_clean, patience=patience)
        calibrator.save(f"checkpoints/PTS/{get_time_str()}")
        save_path = os.path.join("plots/pts_testing", f"joint_pts_{loss_fn}_{get_time_str()}.png")
        calibrator.plot_epoch_losses(save_path=save_path)
        clean_duo_probs = calibrator.calibrate(logits_s=zt_small_clean, logits_l=zt_large_clean)
        clean_metrics = get_metrics_dict(clean_duo_probs, labels_clean)
        log_event(f"Metrics on clean data for JointPTS {loss_fn}: {clean_metrics}")

        duo_probs = calibrator.calibrate(logits_s=zt_small, logits_l=zt_large)
        metrics = get_metrics_dict(duo_probs, labels)
        log_event(f"Metrics on OOD data for JointPTS {loss_fn}: {metrics}")

if __name__ == "__main__":
    main()