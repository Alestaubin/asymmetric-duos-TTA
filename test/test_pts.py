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
    for loss_fn in ["mse", "nll"]:
        calibrator = JointPTS_calibrator(epochs=epochs,
                                        lr=lr,
                                        weight_decay=0.0,
                                        batch_size=bs,
                                        nlayers=2,
                                        n_nodes=256,
                                        length_logits=zt_small.shape[1],
                                        top_k_logits=10,
                                        loss_fn=loss_fn)

        calibrator.tune(logits_s=zt_small_clean, logits_l=zt_large_clean, labels=labels_clean)
        save_path = os.path.join("plots/pts_testing", f"joint_pts_{loss_fn}_{get_time_str()}.png")
        calibrator.plot_epoch_losses(save_path=save_path)
        duo_probs = calibrator.calibrate(logits_s=zt_small, logits_l=zt_large)
        metrics = get_metrics_dict(duo_probs, labels)
        log_event(f"Metrics for JointPTS {loss_fn}: {metrics}")
    # for loss_fn in ["mse", "nll"]:
    #     save_path = os.path.join("plots/pts_testing", f"joint_pts_{loss_fn}_{get_time_str()}.png")
    #     pts_model, loss = get_joint_pts_model(
    #             small_model=small_model,
    #             large_model=large_model,
    #             data_path=data_path,
    #             val_path=None,
    #             epochs=epochs, 
    #             lr=lr,
    #             batch_size=bs,
    #             num_workers=num_workers,
    #             loss_type=loss_fn,
    #             patience=None,
    #             plot_save_path=save_path
    #         )
    #     log_event(f"Trained JointPTS with loss_fn={loss_fn}. Validation Loss: {loss:.6f}")
    #     zd_pts = get_joint_pts_logits(pts_model, zt_large, zt_small)
    #     metrics = get_metrics_dict(F.softmax(zd_pts, dim=-1), labels)
    #     log_event(f"Metrics for JointPTS with loss_fn={loss_fn}: {metrics}")

if __name__ == "__main__":
    main()