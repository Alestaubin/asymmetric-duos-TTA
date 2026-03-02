import os
import torch
import torch.nn as nn
import dependencies.tent.tent as tent
import torch.optim as optim
import torchvision.transforms as trn
import torchvision.datasets as dset

from src.utils.load_utils import pickle_cache
from src.utils.log_utils import log_event
from src.calibration.pts import PTSWrapper, get_pts_model
from src.models.model_loader import get_model
from src.models.inference import get_model_logits_imagenet_c

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
                           steps=int(cfg["OPTIM"]["STEPS"]),
                           episodic=cfg["MODEL"]["EPISODIC"])
    # print(f"model for adaptation: {model}")
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
    print(f"Setting up optimizer: {cfg['OPTIM']['METHOD']} with LR={cfg['OPTIM']['LR']} and Steps={cfg['OPTIM']['STEPS']}")
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

# @pickle_cache("tent_logits_trajectory_cache")
def get_tent_logits_imagenet_c(model_name, 
                                distortion, 
                                severity, 
                                data_path, 
                                cfg: dict,
                                ts=None):
    """
    Caches the adaptation trajectory for a SPECIFIC list of severities.
    """
    assert ts in [None, "pts", 'naive']

    episodic = cfg["TENT"]["MODEL"]["EPISODIC"]
    log_event(f"Running TENT on {model_name} with distortion={distortion} | Severity={severity} | Episodic={episodic} | TS={ts}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fresh = get_model(model_name, freeze = False)

    if ts == "pts":
        pts_model = get_pts_model(model_name=model_name, 
                                        data_path=cfg['val_path'], 
                                        epochs=cfg['PTS']['SINGLE']['epochs'], 
                                        lr=cfg['PTS']['SINGLE']['lr'], 
                                        batch_size=cfg['PTS']['SINGLE']['batch_size'])
        # NOTE: when running tent on the following PTS-wrapped model, does the PTS model also get updated during tent adaptation? 
        # If so, is this desirable or should we freeze the PTS model during tent adaptation?
        model_fresh = PTSWrapper(model_fresh, pts_model)
    elif ts == "naive":
        # print("Applying Naive Temperature Scaling to TENT model...")
        from src.calibration.temperature import TemperatureWrapper
        # get the temperature using the clean validation set
        from src.calibration.temperature import calibrate_temperature
        # Load clean validation logits for this model
        zl_val, labels_val = get_model_logits_imagenet_c(model_name = model_name, 
                                                            distortion="none", 
                                                            severity=0, 
                                                            data_path=cfg["val_path"], 
                                                            batch_size=cfg["TENT"]["TEST"]["BATCH_SIZE"], 
                                                            num_workers=cfg["TENT"]["TEST"]["WORKERS"], 
                                                            split="val")
        optimal_T = calibrate_temperature(zl_val, labels_val)
        model_fresh = TemperatureWrapper(model_fresh, temperature=optimal_T)

    tented_model = setup_tent(model_fresh, cfg["TENT"])
    tented_model = tented_model.to(device)

    preprocess = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # tented_model.reset() 

    if distortion == "none" and severity == 0:
        root_path = data_path # Assumes path to clean val images
    else:
        root_path = os.path.join(data_path, distortion, str(severity))
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Warning: Path {root_path} not found.")
        
    dataset = dset.ImageFolder(root=root_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["TENT"]["TEST"]["BATCH_SIZE"], 
        num_workers=cfg["TENT"]["TEST"]["WORKERS"], pin_memory=True
    )

    sev_logits = []
    sev_labels = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        with torch.enable_grad():
            logits = tented_model(data)
        
        sev_logits.append(logits.detach().cpu())
        sev_labels.append(target.cpu())

    return torch.cat(sev_logits), torch.cat(sev_labels)

