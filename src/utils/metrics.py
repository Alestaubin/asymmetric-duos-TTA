# from https://github.com/tochris/pts-uncertainty/blob/main/metrics.py

import numpy as np
import calibration as cal
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, log_loss, roc_auc_score
import torch
import numpy as np


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
    ece = cal.get_ece(probs, labels, num_bins=15)

    metrics = {}
    metrics['ece'] = ece
    metrics['nll'] = nll
    metrics['accuracy'] = acc
    metrics['f1'] = f1
    metrics['brier'] = brier
    return metrics

