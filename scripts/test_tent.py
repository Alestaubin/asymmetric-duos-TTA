import torch
import torch.nn.functional as F
import numpy as np
from time import time
from src.utils.metrics import get_metrics_dict
from src.utils.load_utils import load_config
from src.tta.tent_utils import get_tent_logits_imagenet_c
from src.utils.log_utils import log_event
from src.models.inference import get_model_logits_imagenet_c
import dependencies.tent.tent as tent
import torch.optim as optim
import torchvision.models as models

def get_entropy(logits: torch.Tensor) -> float:
    """Calculates the average entropy of a batch of logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
    return entropy.mean().item()

def main():
    # 1. Setup
    config = load_config("cfgs/get_metrics.yaml")
    
    model_name = config['large_model']
    # Use a single, high-impact distortion for a clear signal
    distortion = "gaussian_noise"
    severity = 3 
    
    log_event("="*60)
    log_event(f"INVESTIGATION: TENT vs TENT+TS divergence")
    log_event(f"Model: {model_name} | Target: {distortion} Sev {severity}")
    log_event("="*60)

    # log_event(">>> Running base model without tent...")
    # # Get the base model logits for the target distribution (no adaptation)
    # logits_base, labels = get_model_logits_imagenet_c(
    #     model_name=model_name, distortion=distortion, severity=severity, 
    #     data_path=config['data_path']
    # )
    # metrics_base = get_metrics_dict(F.softmax(logits_base, dim=-1), labels)
    # print(metrics_base)
    # acc_base = (logits_base.max(1)[1] == labels).float().mean().item()
    # ent_base = get_entropy(logits_base)

    # model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # model = tent.configure_model(model)
    # params, param_names = tent.collect_params(model)

    # optimizer =  optim.Adam(params,
    #                 lr=0.0001,
    #                 betas=(0.9, 0.999),
    #                 weight_decay=0.0)
    
    # # 4. Wrap in TENT
    # tent_model = tent.Tent(model, optimizer,
    #                        steps=1,
    #                        episodic=True)

    # print(f"model for adaptation: {model}")
    print(f"params for adaptation: {param_names}")
    print(f"optimizer for adaptation: {optimizer}")

    # ---------------------------------------------------------
    # TEST 1: TENT WITHOUT TS
    # ---------------------------------------------------------
    log_event(">>> Running TENT (Standard)...")
    start = time()
    # We use a dummy ts value or force a cache skip if necessary to ensure fresh results
    logits_raw_dict, labels_dict = get_tent_logits_imagenet_c(
        model_name, distortion, [severity], config['data_path'], config, ts=None
    )
    logits_raw = logits_raw_dict[severity]
    labels = labels_dict[severity]
    acc_raw = (logits_raw.max(1)[1] == labels).float().mean().item()
    ent_raw = get_entropy(logits_raw)
    metrics_raw = get_metrics_dict(F.softmax(logits_raw, dim=-1), labels)
    print(metrics_raw)
    log_event(f"Standard TENT -> Acc: {acc_raw:.4f} | Avg Entropy: {ent_raw:.4f} | Time: {time()-start:.2f}s")

    # ---------------------------------------------------------
    # TEST 2: TENT WITH NAIVE TS
    # ---------------------------------------------------------
    log_event(">>> Running TENT (Naive TS)...")
    start = time()
    logits_ts_dict, _ = get_tent_logits_imagenet_c(
        model_name, distortion, [severity], config['data_path'], config, ts="naive"
    )
    logits_ts = logits_ts_dict[severity]
    acc_ts = (logits_ts.max(1)[1] == labels).float().mean().item()
    ent_ts = get_entropy(logits_ts)
    metrics_ts = get_metrics_dict(F.softmax(logits_ts, dim=-1), labels)
    print(metrics_ts)
    log_event(f"TS-TENT       -> Acc: {acc_ts:.4f} | Avg Entropy: {ent_ts:.4f} | Time: {time()-start:.2f}s")

    # ---------------------------------------------------------
    # THE VERDICT
    # ---------------------------------------------------------
    log_event("="*60)
    diff = torch.abs(logits_raw - logits_ts).sum().item()
    
    if diff == 0:
        log_event("CRITICAL: The logits are IDENTICAL to the last decimal.")
        log_event("Check: Is the pickle_cache hashing the 'ts' argument correctly?")
    elif acc_raw == acc_ts:
        log_event("DIVERGENCE DETECTED: Logits differ, but accuracy is identical.")
        log_event(f"L1 Logit Difference: {diff:.4f}")
        log_event("Conclusion: TS is working, but it's not strong enough to flip predictions yet.")
    else:
        log_event("SUCCESS: Adaptation paths have diverged.")
        log_event(f"Accuracy Delta: {abs(acc_raw - acc_ts):.4f}")
    log_event("="*60)

if __name__ == "__main__":
    main()