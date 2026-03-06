
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.inference import get_model_logits_imagenet_c
from src.utils.load_utils import pickle_cache
import copy
from src.utils.plot_utils import plot_epoch_losses, get_time_str
from src.utils.log_utils import log_event
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

        
class JointPTS(nn.Module):
    def __init__(self, num_classes=1000, hidden_dim=256):
        super(JointPTS, self).__init__()
        # Input dimension is the sum of both full logit vectors
        input_dim = num_classes * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2), # Outputs [T_l, T_s]
            nn.Softplus()
        )

    def forward(self, zl, zs):
        # Concatenate logit tuples 
        x = torch.cat([zl, zs], dim=1)
        temps = self.net(x)
        # Add a small epsilon (1e-6) to avoid exactly zero temperatures
        Tl = temps[:, 0:1] + 1e-6
        Ts = temps[:, 1:2] + 1e-6
        return Tl, Ts

class PTS(nn.Module):
    def __init__(self, num_classes=1000, hidden_dim=256):
        super(PTS, self).__init__()
        input_dim = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), # Outputs T
            nn.Softplus()                 # Ensures T > 0
        )

    def forward(self, x):
        temps = self.net(x)
        # Add a small epsilon (1e-6) to avoid exactly zero temperatures
        T = temps[:, 0:1] + 1e-6
        return T

class PTSWrapper(nn.Module):
    """
    Wraps a base model with a PTS model to perform temperature scaling on the logits.
        - The PTS model takes the raw logits as input and outputs a temperature T.
        - The base model's logits are then divided by T to produce calibrated logits.
    Formula: scaled_logits = logits / T
    """
    def __init__(self, base_model, pts_model):
        super(PTSWrapper, self).__init__()
        self.base_model = base_model
        self.pts_model = pts_model

    def forward(self, x):
        logits = self.base_model(x)
        T = self.pts_model(logits)
        scaled_logits = logits / T
        return scaled_logits

# @pickle_cache("joint_pts_model_cache")
def get_joint_pts_model(small_model, 
                        large_model, 
                        data_path, 
                        val_path=None, 
                        epochs=50, 
                        lr=1e-4, 
                        batch_size=128, 
                        num_workers=8, 
                        loss_type="mse",
                        patience=None, 
                        plot_save_path=None):
    """
    Trains JointPTS with Early Stopping.
    Supports routing validation to a proxy shift to prevent clean-data overfitting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1000 
    
    # 1. Fetch Clean Training Logits
    logits_l, labels = get_model_logits_imagenet_c(large_model, "none", 0, data_path, batch_size=batch_size, num_workers=num_workers)
    logits_s, _      = get_model_logits_imagenet_c(small_model, "none", 0, data_path, batch_size=batch_size, num_workers=num_workers)
    
    labels = labels.long()

    if val_path is None:
        log_event("No separate validation path provided. Using random split of training logits for validation.")
        full_dataset = TensorDataset(logits_l, logits_s, labels)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        # Load the proxy shift (e.g., Gaussian Noise) for robust validation
        logits_l_val, labels_val = get_model_logits_imagenet_c(large_model, "gaussian_noise", 3, val_path, batch_size=batch_size, num_workers=num_workers)
        logits_s_val, _          = get_model_logits_imagenet_c(small_model, "gaussian_noise", 3, val_path, batch_size=batch_size, num_workers=num_workers)
        labels_val = labels_val.long()
        
        train_dataset = TensorDataset(logits_l, logits_s, labels)
        val_dataset = TensorDataset(logits_l_val, logits_s_val, labels_val)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = JointPTS(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "nll":
        # CrossEntropyLoss expects raw logits and 1D integer targets
        # It applies LogSoftmax and NLL internally for stability
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function. Choose 'mse' or 'nll'.")  
    
    # Early Stopping Variables
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    print(f"--- Training Full Joint PTS | Max Epochs: {epochs} | Loss: {loss_type.upper()} ---")
    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        for b_zl, b_zs, b_labels in train_loader:
            b_zl, b_zs, b_labels = b_zl.to(device), b_zs.to(device), b_labels.to(device)
            optimizer.zero_grad()
            
            Tl, Ts = model(b_zl, b_zs)
            joint_logits = 0.5 * ((b_zl / Tl) + (b_zs / Ts))
            
            if loss_type == "mse":
                probs = F.softmax(joint_logits, dim=1)
                labels_one_hot = F.one_hot(b_labels, num_classes=num_classes).float()
                loss = loss_fn(probs, labels_one_hot)
            elif loss_type == "nll":
                loss = loss_fn(joint_logits, b_labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_zl, b_zs, b_labels in val_loader:
                b_zl, b_zs, b_labels = b_zl.to(device), b_zs.to(device), b_labels.to(device)
                Tl, Ts = model(b_zl, b_zs)
                joint_logits = 0.5 * ((b_zl / Tl) + (b_zs / Ts))
                
                if loss_type == "mse":
                    probs = F.softmax(joint_logits, dim=1)
                    labels_one_hot = F.one_hot(b_labels, num_classes=num_classes).float()
                    loss = loss_fn(probs, labels_one_hot)
                elif loss_type == "nll":
                    loss = loss_fn(joint_logits, b_labels)
                    
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        epoch_losses_train.append(avg_train_loss)
        epoch_losses_val.append(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if patience is not None and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_loss:.6f}")
            break

    if plot_save_path is not None:
        # plot the epoch loss for both 
        plot_epoch_losses(train_losses=epoch_losses_train, val_losses=epoch_losses_val, save_path=plot_save_path)

    # Load the best weights back into the model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return model.eval(), best_loss

@pickle_cache("pts_model_cache")
def get_pts_model(model_name, data_path, epochs=50, lr=1e-4, batch_size=128, num_workers=8):
    """
    Trains PTS using only the large model logits and Squared Error Loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1000 # Assume ImageNet 1000 classes
    
    logits, labels = get_model_logits_imagenet_c(model_name, "none", 0, data_path, batch_size=batch_size, num_workers=num_workers)
    
    # Prepare one-hot labels for the indicator function I_nc
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    
    logits = logits.to(device)
    
    dataset = TensorDataset(logits, labels_one_hot)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PTS(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"--- Training PTS | Input Dim: {num_classes} ---")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for b_zl, b_labels in loader:
            optimizer.zero_grad()
            
            T = model(b_zl)
            
            # Compute Scaled Logits as per equation: zl / T
            scaled_logits = b_zl / T
            
            # 3. Softmax probabilities
            probs = F.softmax(scaled_logits, dim=1)
            
            # Squared Error Loss L_theta
            # L = 1/N * sum_n( sum_c( (I_nc - sigma_SM_nc)^2 ) )
            loss = torch.sum((b_labels - probs)**2, dim=1).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Squared Error Loss: {total_loss/len(loader):.6f}")
            
    return model.eval()

def get_pts_logits(model, logits):
    """
    Passes the raw logits through the PTS model to get calibrated logits.
    """
    device = next(model.parameters()).device
    logits = logits.to(device)
    with torch.no_grad():
        T = model(logits)
        scaled_logits = logits / T
    return scaled_logits.cpu()


def get_joint_pts_logits(model, zl, zs):
    """
    Passes the raw logit tuples through the JointPTS model to get calibrated logits.
    """
    device = next(model.parameters()).device
    zl, zs = zl.to(device), zs.to(device)
    with torch.no_grad():
        Tl, Ts = model(zl, zs)
        joint_logits = 0.5 * ( (zl / Tl) + (zs / Ts) )
    return joint_logits.cpu()