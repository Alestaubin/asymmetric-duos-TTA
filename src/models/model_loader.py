import torch
import torchvision.models as models
import os
import sys
from src.tta.tent_utils import setup_tent

sys.path.append(os.path.join(os.path.dirname(__file__), '../../dependencies/tent'))
try:
    import tent
except ImportError:
    print("Warning: TENT repository not found in dependencies/. TTA features will be disabled.")

def get_model(model_name, freeze=True, tent_enabled=False, cfg=None):
    """
    Dispatcher function to load models by name.
    """
    model_name = model_name.lower().strip()
    
    if "resnet50" in model_name:
        model = load_resnet50()
    elif "convnext_base" in model_name:
        model = load_convnext_base()
    else:
        raise ValueError(f"Model {model_name} not recognized. Add it to load_models.py")

    if tent_enabled:
        print(f"Configuring {model_name} with TENT (TTA)...")
        assert cfg is not None, "Config (cfg) must be provided when tent_enabled=True"
        model = setup_tent(model, cfg)
        
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model

def load_resnet50():
    """
    Loads ResNet-50 pretrained on ImageNet-1K.
    """
    print("Loading ResNet-50 (Pretrained: ImageNet-1K)...")
    # Using the modern 'weights' argument (Torchvision >= 0.13)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval() # Set to evaluation mode for calibration/TTA
    return model

def load_convnext_base():
    """
    Loads ConvNeXt-Base pretrained on ImageNet-1K.
    """
    print("Loading ConvNeXt-Base (Pretrained: ImageNet-1K)...")
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.eval()
    return model