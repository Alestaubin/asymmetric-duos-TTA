import torch
import torchvision.models as models
import os
import sys
from src.utils.log_utils import log_event

def get_model(model_name, freeze=True):
    """
    Dispatcher function to load models by name.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "resnet50":
        model = load_resnet50()
    elif model_name == "convnext_base":
        model = load_convnext_base()
    elif model_name == "resnet18":
        model = load_resnet18()
    elif model_name == "resnet34":
        model = load_resnet34()
    elif model_name == "wide_resnet50_2":
        model = load_wideresnet50_2()
    else:
        raise ValueError(f"Model {model_name} not recognized. Add it to load_models.py")
        
    if freeze:
        log_event(f"WARNING: Freezing all parameters in {model_name}...")
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

def load_resnet18():
    """
    Loads ResNet-18 pretrained on ImageNet-1K.
    """
    print("Loading ResNet-18 (Pretrained: ImageNet-1K)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def load_resnet34():
    """
    """
    print("Loading ResNet-34 (Pretrained: ImageNet-1K)...")
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.eval()
    return model

    
def load_convnext_base():
    """
    Loads ConvNeXt-Base pretrained on ImageNet-1K.
    """
    print("Loading ConvNeXt-Base (Pretrained: ImageNet-1K)...")
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def load_wideresnet50_2():
    """
    Loads Wide ResNet-50-2 pretrained on ImageNet-1K.
    """
    print("Loading Wide ResNet-50-2 (Pretrained: ImageNet-1K)...")
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.eval()
    return model