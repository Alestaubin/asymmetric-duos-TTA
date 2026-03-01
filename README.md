# Dynamic Duos for Duos that update
A short project that extends Asymmetric Duos by making them more robust to distribution shifts through parametrized temperature scaling and test-time adaptation

Currently exploring the following models:
| Model | GFLOPS |  acc@1 ImageNet-1K | Link | 
| --- | --- | --- | --- |
| `Convnext-b` | 15.36 | 84.062|[Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html#convnext-base)|
| `ResNet50` | 4.09 | 76.13 | [Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#resnet50)| 
| `ResNet18` | 1.81 | 69.758| [Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#resnet18)|
| `ResNet34` | 3.66 | 73.314| [Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html#resnet34)|
| `Wide ResNet50`| 11.40 |78.468 | [Torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html#wide-resnet50-2)|

## Tent Hyperparams

## Directory structure
```
asymmetric-duos-tta/
├── data/                   # (Symlinked)
│   ├── imagenet/
│   └── imagenet-c/
├── dependencies/   
│   └── tent/               # repository from the paper TENT
├── results/                # Output of experiments 
│   ├── metrics/            # results of 2_metrics.py
│   └── TODO/              
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

