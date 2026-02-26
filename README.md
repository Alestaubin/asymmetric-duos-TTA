# Dynamic Duos for Duos that update
A short project that extends Asymmetric Duos by making them more robust to distribution shifts through parametrized temperature scaling and test-time adaptation

Currently exploring the following models:
- `Convnext-b`, 15.36 GFlops
- `ResNet50`, 4.09 GFlops
- `ResNet18`, 1.81 GFlops

## Directory structure
```
asymmetric-duos-tta/
├── data/                   # (Symlinked)
│   ├── imagenet/
│   └── imagenet-c/
├── dependencies/   
│   └── tent/               # repository from the paper TENT
├── checkpoints/            # Pre-trained backbones 
├── results/                # Output of experiments 
│   ├── phase1_baselines/   # CSVs with Top-1 and ECE
│   ├── phase2_entropy/     # JSONs of entropy distributions
│   └── plots/              # Reliability diagrams and Histograms
├── src/                    
│   ├── models/             # Model architectures and Duo wrapper
│   ├── calibration/        # Temperature scaling stuff
│   ├── tta/                # TENT adaptation 
│   └── utils/              # ECE calculator, ImageNet-C loader
├── scripts/                
│   ├── 1_save_logits.py    # save the logits of all models on ImageNet-C
│   ├── 2_metrics.py        # Compute metrics for all model configurations from the saved logits
│   └── 3_TODO.py
├── requirements.txt
└── main.py                 # Primary entry point for evaluations
```


When this repository is cloned, `dependencies/tent` will be empty. You must run:
```bash
git submodule update --init --recursive
```

