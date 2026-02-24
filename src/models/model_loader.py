import torch
import torchvision.models as models

def get_model(model_name, freeze=True):
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