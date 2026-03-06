import matplotlib.pyplot as plt
# import scienceplots
import os
import time
# plt.style.use(['ieee', 'science'])

def get_time_str():
    """Get current time as a formatted string."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def plot_epoch_losses(train_losses, val_losses, save_path=None):
    """
    Plots training and validation losses across epochs.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.title('Epoch Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Epoch loss plot saved to {save_path}")
    
    plt.show()