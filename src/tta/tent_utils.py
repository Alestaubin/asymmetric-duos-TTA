import torch
import dependencies.tent.tent as tent
from src.utils.load_utils import pickle_cache
import torch.optim as optim

def setup_tent(model, cfg):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    print(f"model for adaptation: {model}")
    print(f"params for adaptation: {param_names}")
    print(f"optimizer for adaptation: {optimizer}")
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
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

@pickle_cache("tent_logits_cache")
def get_tent_logits_imagenet_c(tented_model, distortion_name, severity, data_path, batch_size, num_workers, split="test"):
    """
    Performs Test-Time Adaptation (TTA) using TENT.
    Returns the logits produced during the adaptation process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tented_model = tented_model.to(device)
    # Setup data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    
    if split == "val":
        root_path = data_path # Assumes path to clean val images
    else:
        root_path = os.path.join(data_path, distortion_name, str(severity))
        
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         num_workers=num_workers, pin_memory=True)

    tented_model.eval()  # Note: TENT uses model.train() internally for BN updates
    all_logits = []
    all_labels = []

    print(f"Starting TTA Adaptation for {distortion_name} (Severity {severity})...")

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

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