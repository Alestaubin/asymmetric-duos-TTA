# from https://github.com/tochris/pts-uncertainty/blob/main/metrics.py

import numpy as np
import calibration as cal
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score
import torch
import numpy as np

def ece(probs, labels, num_bins=15):
    """
    Expected calibration error

    probs: np.array of shape (num_samples, num_classes) - predicted probabilities
    labels: np.array of shape (num_samples). labels[i] denotes the label of the i-th
            example
    num_bins: int - number of bins to use for ECE calculation
    """
    ece = cal.get_ece(probs, labels, num_bins=num_bins)
    #[lower, ece, upper] = cal.get_top_calibration_error(probs, labels, p=1)
    # when p=1, mid is the ECE
    return ece

def tim_ece(probs, labels, n_bins=15):
    """Computes Expected Calibration Error (ECE)."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_confidence - bin_accuracy)
    return ece.item()


def get_metrics_dict(probs, labels) -> dict:
    """
    Computes ECE, NLL, and Accuracy for the given probabilities and labels.
    
    probs: np.array of shape (num_samples, num_classes) - predicted probabilities
    labels: np.array of shape (num_samples). labels[i] denotes the label of the i-th
            example
    """
    # Check that the probs sum to 1 for each sample
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6), "Probabilities must sum to 1 for each sample."
    
    num_classes = probs.shape[1]
    
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    brier = brier_score_loss(
        y_true=np.eye(num_classes)[labels].reshape(-1),
        y_proba=probs.reshape(-1)
    )
    nll = log_loss(labels, probs, labels=list(range(num_classes)))
    ece1 = tim_ece(torch.tensor(probs), torch.tensor(labels))
    ece2 = ece(probs, labels)


    metrics = {}
    metrics['tim_ece'] = ece1
    metrics['ece'] = ece2
    metrics['nll'] = nll
    metrics['accuracy'] = acc
    metrics['f1'] = f1
    metrics['brier'] = brier
    return metrics

