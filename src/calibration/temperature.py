import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from src.utils.load_utils import pickle_cache


def jointly_calibrate_temperature(logits_l, logits_s, labels):
    '''
    Taken from https://github.com/timgzhou/asymmetric-duos/blob/main/evaluate/3_duo_temp_scale.py
    '''
    print("=====Joint temperature calibration in progress...=====")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logits_l = logits_l.to(device)
    logits_s = logits_s.to(device)
    labels = labels.to(device)
    
    best_nll = float("inf")
    best_Tl, best_Ts = 1.0, 1.0

    for Tl in torch.arange(0.05, 5.05, 0.2):
        for Ts in torch.arange(0.05, 5.05, 0.2):
            logits_avg = (logits_l / Tl + logits_s / Ts) / 2
            nll = F.cross_entropy(logits_avg, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_Tl, best_Ts = Tl.item(), Ts.item()

    print(f"Grid best Tl={best_Tl:.2f}, Ts={best_Ts:.2f}, NLL={best_nll:.4f}")

    Tl = torch.tensor([best_Tl], requires_grad=True, device=logits_l.device)
    Ts = torch.tensor([best_Ts], requires_grad=True, device=logits_s.device)
    optimizer = torch.optim.LBFGS([Tl, Ts], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        logits_avg = (logits_l / Tl + logits_s / Ts) / 2
        loss = F.cross_entropy(logits_avg, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_Tl, final_Ts = Tl.item(), Ts.item()
    print(f"Refined Tl={final_Tl:.4f}, Ts={final_Ts:.4f}")
    print(f"Final NLL = {F.cross_entropy((logits_l / Tl + logits_s / Ts)/2, labels).item():.4f}")
    print(f"Validation accuracy with joint scaling: {((logits_l / Tl + logits_s / Ts)/2).max(1)[1].eq(labels).float().mean().item():.4f}")
    print(f"Validation accuracy of large model: {logits_l.max(1)[1].eq(labels).float().mean().item():.4f}")
    print(f"Validation accuracy of small model: {logits_s.max(1)[1].eq(labels).float().mean().item():.4f}")
    print("=====Joint calibration complete and models wrapped.=====")
    return final_Tl,final_Ts
    
# def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
#     '''
#         Taken from https://github.com/timgzhou/asymmetric-duos/blob/main/evaluate/3_duo_temp_scale.py
#     '''
#     original_nll = F.cross_entropy(logits, labels).item()
#     print(f"NLL before temperature scaling = {original_nll:.4f}")
#     best_nll = float("inf")
#     best_T = 1.0
#     for T in torch.arange(0.05, 5.05, 0.05):
#         T = T.item()
#         loss = F.cross_entropy(logits / T, labels).item()
#         if loss < best_nll:
#             best_nll = loss
#             best_T = T
#     print(f"Grid search best T = {best_T:.3f}, NLL = {best_nll:.4f}")

#     print(f"Use LBFGS to find a fine-grained temperature")
#     temp_tensor = torch.tensor([best_T], requires_grad=True, device=logits.device)
#     optimizer = torch.optim.LBFGS([temp_tensor], lr=0.01, max_iter=50)

#     def closure():
#         optimizer.zero_grad()
#         loss = F.cross_entropy(logits / temp_tensor, labels)
#         loss.backward()
#         return loss

#     optimizer.step(closure)
#     T_refined = temp_tensor.detach().item()

#     final_nll = F.cross_entropy(logits / T_refined, labels).item()
#     print(f"Refined T = {T_refined:.4f}")
#     print(f"NLL after temperature scaling = {final_nll:.4f}")
#     return T_refined

@pickle_cache("calibrated_temperatures")
def calibrate_temperature(logits, labels):
    """
    Finds the temperature that minimizes NLL on clean validation data.
    """
    print("Starting temperature calibration...")  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
    logits = logits.to(device)
    labels = labels.to(device)

    # Optimization loop for T
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5) # Start guess at 1.5
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        # Scale logits by current T
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    optimal_T = temperature.item()
    print(f"Calibration complete. Optimal Temperature T={optimal_T:.4f}")
    return optimal_T

def get_joint_pts_model(logits_l, logits_s, labels):
    """
    This function trains a small neural network to predict the optimal temperatures for the large and small models based on their logits.
    """
    raise NotImplementedError("Joint PTS not yet implemented. Please use calibrate_temperature for now or set ts=None or ts='naive' when extracting TENT logits.")

class JointPTS(torch.nn.Module):
    pass

class TemperatureWrapper(nn.Module):
    """
    Wraps a model to scale logits by a fixed temperature T.
    Formula: logits / T
    """
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        # register buffer so it saves with state_dict but isn't a trainable param
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
