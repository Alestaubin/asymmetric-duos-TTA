import torch
import torch.nn.functional as F


def jointly_calibrate_temperature(logits_l, logits_s, labels):
    '''
    Taken from https://github.com/timgzhou/asymmetric-duos/blob/main/evaluate/3_duo_temp_scale.py
    '''
    print("=====Joint temperature calibration in progress...=====")
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

    print("=====Joint calibration complete and models wrapped.=====")
    return final_Tl,final_Ts


class JointPTS(torch.nn.Module):
    def __init__(self, model_l, model_s, Tl, Ts):
        super(JointPTS, self).__init__()
        self.model_l = model_l
        self.model_s = model_s
        self.Tl = Tl
        self.Ts = Ts

    def forward(self, x):
        logits_l = self.model_l(x) / self.Tl
        logits_s = self.model_s(x) / self.Ts
        return (logits_l + logits_s) / 2