import argparse
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import pandas as pd
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from time import time

from src.models.model_loader import get_model
from src.utils import metrics
from src.utils.load_utils import pickle_cache  # Your decorator

parser = argparse.ArgumentParser(description='Asymmetric Duo Phase 1: Robustness Evaluation')
parser.add_argument('--large-model', '-L', type=str, required=True, help='e.g., convnext_base')
parser.add_argument('--small-model', '-S', type=str, required=True, help='e.g., resnet50')
parser.add_argument('--data-path', type=str, default='./data/imagenet-c', help='Path to ImageNet-C')
parser.add_argument('--val-path', type=str, required=True, help='Path to clean ImageNet-Val for calibration')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--output-dir', type=str, default='./results/phase1_baselines')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# /////////////// Logic for Duo Abstraction ///////////////

@pickle_cache("logits_cache")
def get_model_logits_imagenet_c(model_name, distortion, severity, data_path, split="test"):
    """
    Extracts or loads cached logits for a specific model/distortion combo.
    Note: We pass parameters to the decorator via the function arguments.
    """
    # Load model locally inside function to keep the cache clean
    net = get_model(model_name, freeze=True)
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
    net.eval()

    # Setup data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    
    if split == "val":
        root_path = data_path # Assumes path to clean val images
    else:
        root_path = os.path.join(data_path, distortion, str(severity))
        
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                         num_workers=args.workers, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            output = net(data.cuda())
            all_logits.append(output.cpu())
            all_labels.append(target.cpu())
            
    return torch.cat(all_logits), torch.cat(all_labels)

def jointly_calibrate_temperature(logits_l, logits_s, labels):
    '''
    Taken from https://github.com/timgzhou/asymmetric-duos/blob/main/evaluate/3_duo_temp_scale.py
    '''
    print("=====Joint temperature calibration in progress...=====")
    best_nll = float("inf")
    best_Tl, best_Ts = 1.0, 1.0

    for Tl in torch.arange(0.05, 5.05, 0.2):
        for Ts in torch.arange(0.05, 5.05, 0.2):
            logits_avg = (logits_l / Tl + logits_s / Ts) / 2
            nll = F.cross_entropy(logits_avg, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_Tl, best_Ts = Tl.item(), Ts.item()

    print(f"Grid best Tl={best_Tl:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")

    Tl = torch.tensor([best_Tl], requires_grad=True, device=logits_l.device)
    Ts = torch.tensor([best_Ts], requires_grad=True, device=logits_s.device)
    optimizer = torch.optim.LBFGS([Tl, Ts], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        logits_avg = (logits_l / Tl + logits_s / Ts) / 2
        loss = F.cross_entropy(logits_avg, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_Tl, final_Ts = Tl.item(), Ts.item()
    print(f"Refined Tl={final_Tl:.4f}, Ts={final_Ts:.4f}")
    print(f"Final NLL = {F.cross_entropy((logits_l / Tl + logits_s / Ts)/2, labels).item():.4f}")

    print("=====Joint calibration complete and models wrapped.=====")
    return final_Tl,final_Ts
# /////////////// Execution Pipeline ///////////////

# 1. GET CALIBRATION TEMPERATURES (On Clean Val)
print("\n--- Step 1: Calibration ---")
start_time = time()
zl_val, labels_val = get_model_logits_imagenet_c(args.large_model, "none", 0, args.val_path, split="val")
end_time = time()
print(f"Large model calibration data obtained in {end_time - start_time:.2f} seconds.")

start_time = time()
zs_val, _          = get_model_logits_imagenet_c(args.small_model, "none", 0, args.val_path, split="val")
end_time = time()
print(f"Small model calibration data obtained in {end_time - start_time:.2f} seconds.")

Tl, Ts = jointly_calibrate_temperature(zl_val, zs_val, labels_val)

# 2. RUN EVALUATION ON IMAGENET-C
print("\n--- Step 2: Evaluation on ImageNet-C ---")
distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

all_results = []

for d_name in distortions:
    print(f"Processing {d_name}...")
    for sev in range(1, 6):
        print(f"  Severity {sev}...")
        # The cache handles if we need to run the GPU or just load the pickle
        start_time = time()
        zl, labels = get_model_logits_imagenet_c(args.large_model, d_name, sev, args.data_path)
        end_time = time()
        print(f"Large model logits obtained in {end_time - start_time:.2f} seconds.")
        start_time = time()
        zs, _      = get_model_logits_imagenet_c(args.small_model, d_name, sev, args.data_path)
        end_time = time()
        print(f"Small model logits obtained in {end_time - start_time:.2f} seconds.")
        
        # Duo Fusion Logic
        logits_avg = (zl / Tl + zs / Ts) / 2
        
        err = 1.0 - (logits_avg.max(1)[1].eq(labels).sum().item() / len(labels))
        ece = metrics.ece(logits_avg, labels)
        
        all_results.append({
            'distortion': d_name, 'severity': sev,
            'error_rate': err, 'ece': ece,
            'Tl': Tl, 'Ts': Ts
        })

# 3. SAVE RESULTS
df = pd.DataFrame(all_results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"duo_{args.large_model}_{args.small_model}_{timestamp}.csv"
df.to_csv(os.path.join(args.output_dir, filename), index=False)

print(f"\nPhase 1 Complete. mCE: {df['error_rate'].mean():.4f}, Avg ECE: {df['ece'].mean():.4f}")