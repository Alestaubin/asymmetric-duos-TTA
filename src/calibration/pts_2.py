import os
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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PTSNetwork(nn.Module):
    """The PyTorch Neural Network module for PTS."""
    def __init__(self, length_logits, top_k_logits, nlayers, n_nodes):
        super(PTSNetwork, self).__init__()
        self.top_k_logits = top_k_logits
        
        # Build the dynamic MLP layers
        layers = []
        in_features = top_k_logits
        for _ in range(nlayers):
            layers.append(nn.Linear(in_features, n_nodes))
            layers.append(nn.ReLU())
            in_features = n_nodes
            
        # Final temperature output layer
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, logits):
        # 1. Sort and keep top-k logits
        # torch.topk automatically returns the largest elements sorted descending
        top_k, _ = torch.topk(logits, self.top_k_logits, dim=-1)
        
        # 2. Pass through MLP to get raw temperature
        t = self.net(top_k)
        
        # 3. Absolute value to ensure strictly positive T
        temperature = torch.abs(t)
        
        # 4. Clip temperature and scale logits
        # Clamping prevents division by zero (min) or extreme scaling (max)
        temperature = torch.clamp(temperature, min=1e-12, max=1e12)
        scaled_logits = logits / temperature
        
        # 5. Return softmax probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        return probs
        

class PTS_calibrator():
    """Class for Parameterized Temperature Scaling (PTS) - PyTorch Version"""
    def __init__(
        self,
        epochs,
        lr,
        weight_decay,
        batch_size,
        nlayers,
        n_nodes,
        length_logits,
        top_k_logits
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.length_logits = length_logits
        
        # Automatically select GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, optimizer, and loss
        self.model = PTSNetwork(top_k_logits, nlayers, n_nodes).to(self.device)
        
        # L2 regularization is handled natively by PyTorch's weight_decay parameter in the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss()

    def tune(self, logits, labels, clip=1e2):
        """Tune PTS model"""
        # Convert numpy arrays to tensors if necessary
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)

        assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        assert labels.shape[1] == self.length_logits, "labels need to have same length as length_logits!"

        # Clip logits
        logits = torch.clamp(logits, min=-clip, max=clip)

        # Create DataLoader
        dataset = TensorDataset(logits, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_logits, batch_labels in loader:
                batch_logits = batch_logits.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                
                probs = self.model(batch_logits)
                loss = self.criterion(probs, batch_labels)
                
                loss.backward()
                self.optimizer.step()

    def calibrate(self, logits, clip=1e2):
        """Calibrate logits with PTS model"""
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)

        # assert logits.shape[1] == self.length_logits, "logits need to have same length as length_logits!"
        
        logits = torch.clamp(logits, min=-clip, max=clip).to(self.device)

        self.model.eval()
        with torch.no_grad():
            calibrated_probs = self.model(logits)
            
        return calibrated_probs.cpu().numpy()

    def save(self, path="./"):
        """Save PTS model parameters"""
        if not os.path.exists(path):
            os.makedirs(path)

        filepath = os.path.join(path, "pts_model.pth")
        print("Save PTS model to: ", filepath)
        # Save the state dictionary, which is standard PyTorch practice over saving the full object
        torch.save(self.model.state_dict(), filepath)

    def load(self, path="./"):
        """Load PTS model parameters"""
        filepath = os.path.join(path, "pts_model.pth")
        print("Load PTS model from: ", filepath)
        # map_location ensures it loads correctly even if moving from GPU to CPU
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

class JointPTSNetwork(nn.Module):
    """The PyTorch Neural Network module for PTS."""
    def __init__(self, top_k_logits, nlayers, n_nodes):
        super(JointPTSNetwork, self).__init__()
        self.top_k_logits = top_k_logits
        
        # Build the dynamic MLP layers
        layers = []
        in_features = 2*top_k_logits
        for _ in range(nlayers):
            layers.append(nn.Linear(in_features, n_nodes))
            layers.append(nn.ReLU())
            in_features = n_nodes
            
        # Final temperature output layer
        layers.append(nn.Linear(in_features, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, logits_l, logits_s):
        # 1. Sort and keep top-k logits
        # torch.topk automatically returns the largest elements sorted descending
        top_k_l, _ = torch.topk(logits_l, self.top_k_logits, dim=-1)
        top_k_s, _ = torch.topk(logits_s, self.top_k_logits, dim=-1)

        # Concatenate the top-k logits
        top_k = torch.cat([top_k_l, top_k_s], dim=-1)

        # 2. Pass through MLP to get raw temperature
        temps = self.net(top_k)
        
        # 3. Absolute value to ensure strictly positive T
        temps = torch.abs(temps)
        
        # 4. Clip temperature and scale logits
        # Clamping prevents division by zero (min) or extreme scaling (max)
        temps = torch.clamp(temps, min=1e-12, max=1e12)
        Tl = temps[:, 0:1]
        Ts = temps[:, 1:2]

        scaled_logits_l = logits_l / Tl
        scaled_logits_s = logits_s / Ts
        joint_logits = (scaled_logits_l + scaled_logits_s) / 2
        # 5. Return softmax probabilities
        #probs = F.softmax(joint_logits, dim=-1)
        return joint_logits
    
class JointPTS_calibrator():
    """Class for Parameterized Temperature Scaling (PTS) - PyTorch Version"""
    def __init__(
        self,
        epochs,
        lr,
        weight_decay,
        batch_size,
        nlayers,
        n_nodes,
        length_logits,
        top_k_logits,
        loss_fn="mse"
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.length_logits = length_logits
        self.loss_fn = loss_fn
        
        # Automatically select GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, optimizer, and loss
        self.model = JointPTSNetwork(top_k_logits, nlayers, n_nodes).to(self.device)
        
        # L2 regularization is handled natively by PyTorch's weight_decay parameter in the optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_fn == "nll":
            # CrossEntropyLoss expects raw logits and 1D integer targets
            # It applies LogSoftmax and NLL internally for stability
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Choose 'mse' or 'nll'.")  

    def tune(self, logits_s, logits_l, labels, clip=1e2, patience=7):
        """Tune PTS model"""
        # Convert numpy arrays to tensors if necessary
        if not torch.is_tensor(logits_s):
            logits_s = torch.tensor(logits_s, dtype=torch.float32)
        if not torch.is_tensor(logits_l):
            logits_l = torch.tensor(logits_l, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.loss_fn == "mse":
            labels = F.one_hot(labels, num_classes=logits_l.shape[1]).float()
            assert labels.shape[1] == self.length_logits, "labels need to have same length as length_logits!"

        assert logits_s.shape[1] == self.length_logits, "logits_s need to have same length as length_logits!"
        assert logits_l.shape[1] == self.length_logits, "logits_l need to have same length as length_logits!"

        # Clip logits
        logits_s = torch.clamp(logits_s, min=-clip, max=clip)
        logits_l = torch.clamp(logits_l, min=-clip, max=clip)
        
        full_dataset = TensorDataset(logits_l, logits_s, labels)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


        self.epoch_losses_train = []
        self.epoch_losses_val = []

        # Early Stopping Variables
        best_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):            
            epoch_loss = 0
            self.model.train()
            for batch_logits_s, batch_logits_l, batch_labels in train_loader:
                batch_logits_s = batch_logits_s.to(self.device)
                batch_logits_l = batch_logits_l.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                
                joint_logits = self.model(batch_logits_s, batch_logits_l)
                
                if self.loss_fn == 'mse':
                    probs = F.softmax(joint_logits, dim=-1)
                    loss = self.criterion(probs, batch_labels)
                elif self.loss_fn == 'nll':
                    loss = self.criterion(joint_logits, batch_labels)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # Validation:
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for b_zl, b_zs, b_labels in val_loader:
                    b_zl, b_zs, b_labels = b_zl.to(self.device), b_zs.to(self.device), b_labels.to(self.device)
                    joint_logits = self.model(batch_logits_s, batch_logits_l)
                    
                    if self.loss_fn == 'mse':
                        probs = F.softmax(joint_logits, dim=-1)
                        loss = self.criterion(probs, batch_labels)
                    elif self.loss_fn == 'nll':
                        loss = self.criterion(joint_logits, batch_labels)
                        
                    val_loss += loss.item()
            
            avg_loss_train = epoch_loss / len(train_loader)
            avg_loss_val = val_loss / len(val_loader)

            log_event(f"Epoch {epoch} | train loss: {avg_loss_train} | val loss: {avg_loss_val}")

            self.epoch_losses_train.append(avg_loss_train)
            self.epoch_losses_val.append(avg_loss_val)

            # --- EARLY STOPPING CHECK ---
            if patience is not None:
                if avg_loss_val < best_loss:
                    best_loss = avg_loss_val
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    log_event(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.6f}")
                    break

    def plot_epoch_losses(self, save_path=None):
        """
        Plots training and validation losses across epochs.
        
        Args:
            train_losses (list): List of training losses per epoch.
            val_losses (list): List of validation losses per epoch.
            save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses_train, label='Train Loss', marker='o')
        plt.plot(self.epoch_losses_val, label='Val Loss', marker='o')
        plt.title('Epoch Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Epoch loss plot saved to {save_path}")
        
        plt.show()



    def calibrate(self, logits_s, logits_l, clip=1e2):
        """Calibrate logits with PTS model"""
        if not torch.is_tensor(logits_s):
            logits_s = torch.tensor(logits_s, dtype=torch.float32)
        if not torch.is_tensor(logits_l):
            logits_l = torch.tensor(logits_l, dtype=torch.float32)

        assert logits_s.shape[1] == self.length_logits, "logits_s need to have same length as length_logits!"
        assert logits_l.shape[1] == self.length_logits, "logits_l need to have same length as length_logits!"

        logits_s = torch.clamp(logits_s, min=-clip, max=clip).to(self.device)
        logits_l = torch.clamp(logits_l, min=-clip, max=clip).to(self.device)

        self.model.eval()
        with torch.no_grad():
            calibrated_logits = self.model(logits_s, logits_l)
            calibrated_probs = F.softmax(calibrated_logits, dim=-1)

        return calibrated_probs.cpu().numpy()

    def save(self, path="./"):
        """Save PTS model parameters"""
        if not os.path.exists(path):
            os.makedirs(path)

        filepath = os.path.join(path, "pts_model.pth")
        print("Save PTS model to: ", filepath)
        # Save the state dictionary, which is standard PyTorch practice over saving the full object
        torch.save(self.model.state_dict(), filepath)

    def load(self, path="./"):
        """Load PTS model parameters"""
        filepath = os.path.join(path, "pts_model.pth")
        print("Load PTS model from: ", filepath)
        # map_location ensures it loads correctly even if moving from GPU to CPU
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
