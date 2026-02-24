import torch
from src.utils.load_utils import pickle_cache

@pickle_cache("tent_logits_cache")
def run_tta_inference(tented_model, distortion_name, severity, loader):
    """
    Performs Test-Time Adaptation (TTA) using TENT.
    Returns the logits produced during the adaptation process.
    """
    tented_model.eval()  # Note: TENT uses model.train() internally for BN updates
    all_logits = []
    all_labels = []

    print(f"Starting TTA Adaptation for {distortion_name} (Severity {severity})...")

    # TENT adapts by looking at batches of data. 
    # The order of the loader matters here!
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()

        # TENT forward pass:
        # 1. Calculates entropy of predictions
        # 2. Performs backward pass on BN parameters
        # 3. Returns the logits for the current batch
        with torch.enable_grad():  # TENT requires grads to update BN params
            logits = tented_model(data)
        
        # We store the logits produced *during* adaptation to see 
        # how the model performs "on-the-fly".
        all_logits.append(logits.detach().cpu())
        all_labels.append(target.cpu())

        if batch_idx % 10 == 0:
            acc = (logits.detach().max(1)[1] == target).float().mean().item()
            print(f"Batch {batch_idx:03d} | Batch Acc: {acc:.2f}", end="\r")

    print("\nAdaptation for current severity complete.")
    return torch.cat(all_logits), torch.cat(all_labels)