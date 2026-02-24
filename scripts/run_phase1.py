import argparse
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
import numpy as np
import pandas as pd  # Added for CSV handling
from datetime import datetime
from src.models.model_loader import get_model
from src.utils import metrics

parser = argparse.ArgumentParser(description='Evaluates robustness on ImageNet-C')
parser.add_argument('--model-name', '-m', type=str, required=True, help='resnet50 or convnext_base')
parser.add_argument('--data-path', type=str, default='./data/imagenet-c', help='Path to symlinked ImageNet-C')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--output-dir', type=str, default='./results/phase1_baselines', help='Directory to save CSV')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# /////////////// Model Setup ///////////////
net = get_model(args.model_name, freeze=True)

if torch.cuda.is_available():
    net = net.cuda()
    net = torch.nn.DataParallel(net)

net.eval()
print(f'Model {args.model_name} Loaded and Frozen.')
print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")

# /////////////// Data Setup ///////////////
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# /////////////// Evaluation Function ///////////////
def evaluate_distortion(distortion_name):
    # We'll store results in a list of dicts for easy DataFrame conversion
    results = []

    for severity in range(1, 6):
        distorted_path = os.path.join(args.data_path, distortion_name, str(severity))
        dataset = dset.ImageFolder(root=distorted_path, transform=preprocess)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.workers, pin_memory=True)

        correct = 0
        all_logits = []
        all_labels = []

        print(f"Evaluating {distortion_name} - Severity {severity}...", end="\r")
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                output = net(data)
                
                pred = output.max(1)[1]
                correct += pred.eq(target).sum().item()
                
                all_logits.append(output.cpu())
                all_labels.append(target.cpu())

        err = 1.0 - (correct / len(dataset))
        probs = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        ece = metrics.ece(probs, labels)
        
        results.append({
            'distortion': distortion_name,
            'severity': severity,
            'error_rate': err,
            'ece': ece
        })

    return results

# /////////////// Execution ///////////////
print(f"{'Distortion':<20} | {'Avg Error (%)':<15} | {'Avg ECE':<10}")
print("-" * 50)

all_results = []

for d_name in distortions:
    dist_results = evaluate_distortion(d_name)
    all_results.extend(dist_results)
    
    # Calculate average for printing
    avg_err = np.mean([r['error_rate'] for r in dist_results])
    avg_ece = np.mean([r['ece'] for r in dist_results])
    print(f"{d_name:<20} | {100*avg_err:>14.2f}% | {avg_ece:>9.4f}")

# /////////////// Saving Results ///////////////
df = pd.DataFrame(all_results)

# Generate a clean filename: model_dataset_timestamp.csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"{args.model_name}_imagenet_c_{timestamp}.csv"
save_path = os.path.join(args.output_dir, filename)

df.to_csv(save_path, index=False)

print("-" * 50)
print(f"Results saved to: {save_path}")
print(f"{'mCE (Overall)':<20} | {100*df['error_rate'].mean():>14.2f}% | {df['ece'].mean():>9.4f}")