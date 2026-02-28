import torch
import dependencies.tent.tent as tent
from src.utils.load_utils import pickle_cache
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset
import os
import torch.nn as nn
from src.utils.log_utils import log_event
# from src.models.inference import get_model_logits_imagenet_c


def setup_tent(model, cfg):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """

    # 1. Disable all grads first
    model.requires_grad_(False)
    
    # 2. Enable grads for normalization layers (BN for ResNet, LN for ConvNeXt)
    # This replaces tent.configure_model(model) logic to be more inclusive
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            # Force buffers (like running mean/var) to update even in eval mode if needed
            m.track_running_stats = True 
            
    # 3. Collect only those enabled params
    params = []
    param_names = []
    for nm, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            param_names.append(nm)
    
    if not params:
        raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")

    optimizer = setup_optimizer(params, cfg)
    
    # 4. Wrap in TENT
    tent_model = tent.Tent(model, optimizer,
                           steps=int(cfg["OPTIM"]["STEPS"]),
                           episodic=cfg["MODEL"]["EPISODIC"])
    
    log_event(f"Params for adaptation: {len(param_names)}")
    return tent_model


def setup_optimizer(params, cfg):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg["OPTIM"]["METHOD"] == 'Adam':
        return optim.Adam(params,
                    lr=float(cfg["OPTIM"]["LR"]),
                    betas=(float(cfg["OPTIM"]["BETA"]), 0.999),
                    weight_decay=float(cfg["OPTIM"]["WD"]))
    elif cfg["OPTIM"]["METHOD"] == 'SGD':
        return optim.SGD(params,
                   lr=cfg["OPTIM"]["LR"],
                   momentum=cfg["OPTIM"]["MOMENTUM"],
                   dampening=cfg["OPTIM"]["DAMPENING"],
                   weight_decay=cfg["OPTIM"]["WD"],
                   nesterov=cfg["OPTIM"]["NESTEROV"])
    else:
        raise NotImplementedError

@pickle_cache("tent_logits_trajectory_cache")
def get_tent_logits_imagenet_c(model_name, 
                                distortion_name, 
                                severities, 
                                data_path, 
                                tent_cfg: dict, 
                                ts=None):
    print(f"get_tent_logits_imagenet_c called with: model={model_name}, distortion={distortion_name}, severities={severities}, data_path={data_path}, tent_cfg={tent_cfg}, ts={ts}")
    """
    Caches the adaptation trajectory for a SPECIFIC list of severities.
    Preserves model weights across the sequence for continual adaptation.
    """
    from src.models.model_loader import get_model
    import torchvision.datasets as dset
    import torchvision.transforms as trn
    assert ts in [None, "pts", 'naive']
    from src.models.inference import get_model_logits_imagenet_c

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_fresh = get_model(model_name, freeze = False)

    if ts == "pts":
        raise NotImplementedError("PTS temperature scaling not yet implemented for TENT. Please set ts=None or ts='naive'.")
    elif ts == "naive":
        print("Applying Naive Temperature Scaling to TENT model...")
        from src.calibration.temperature import TemperatureWrapper
        # get the temperature using the clean validation set
        from src.calibration.temperature import calibrate_temperature
        # Load clean validation logits for this model
        zl_val, labels_val = get_model_logits_imagenet_c(model_name, "none", 0, tent_cfg["VAL_PATH"], batch_size=tent_cfg["TEST"]["BATCH_SIZE"], num_workers=tent_cfg["TEST"]["WORKERS"], split="val")
        optimal_T = calibrate_temperature(zl_val, labels_val, device=device)
        model_fresh = TemperatureWrapper(model_fresh, temperature=optimal_T)

    tented_model = setup_tent(model_fresh, tent_cfg)
    tented_model = tented_model.to(device)

    trajectory_logits = {}
    trajectory_labels = {}

    preprocess = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Iterate only through the requested severities
    for sev in severities:
        print(f"--- TTA Adaptation: {distortion_name} | Severity {sev} ---")
        
        if distortion_name == "none" and sev == 0:
            root_path = data_path # Assumes path to clean val images
        else:
            root_path = os.path.join(data_path, distortion_name, str(sev))
        
        if not os.path.exists(root_path):
            print(f"Warning: Path {root_path} not found. Skipping.")
            continue

        dataset = dset.ImageFolder(root=root_path, transform=preprocess)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=tent_cfg["TEST"]["BATCH_SIZE"], 
            num_workers=tent_cfg["TEST"]["WORKERS"], pin_memory=True
        )

        sev_logits = []
        sev_labels = []

        # TENT adapts by looking at batches sequentially
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            with torch.enable_grad():
                logits = tented_model(data)
            
            sev_logits.append(logits.detach().cpu())
            sev_labels.append(target.cpu())

        trajectory_logits[sev] = torch.cat(sev_logits)
        trajectory_labels[sev] = torch.cat(sev_labels)

    return trajectory_logits, trajectory_labels