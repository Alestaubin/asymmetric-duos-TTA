# Dynamic Duos for Duos that update

Directory structure: 
```
asymmetric-duos-tta/
├── data/                   # (Symlinked to cluster storage)
│   ├── imagenet/           # Clean ImageNet-Val
│   └── imagenet-c/         # 15 corruptions x 5 severities
├── checkpoints/            # Pre-trained backbones (ViT, ResNet, SatMAE)
├── results/                # Output of experiments (Google Doc data)
│   ├── phase1_baselines/   # CSVs with Top-1 and ECE
│   ├── phase2_entropy/     # JSONs of entropy distributions
│   └── plots/              # Reliability diagrams and Histograms
├── src/                    # The "Engine"
│   ├── models/             # Model architectures and Duo wrapper
│   ├── calibration/        # JointPTSHead and L-BFGS logic
│   ├── tta/                # TENT and EATA adaptation loops
│   └── utils/              # ECE calculator, ImageNet-C loader
├── scripts/                # The "Drivers"
│   ├── run_phase1.py       # Script for ImageNet-C baseline
│   ├── run_phase3_pts.py   # Training script for PTS head
│   └── run_tta_sweep.sh    # Bash script to loop over corruptions
├── environment.yml         # Conda environment config
└── main.py                 # Primary entry point for evaluations
```


When this repository is cloned, `dependencies/tent` will be empty. You must run:
```bash
git submodule update --init --recursive
```