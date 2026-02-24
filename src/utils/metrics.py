# from https://github.com/tochris/pts-uncertainty/blob/main/metrics.py

import numpy as np
import calibration as cal
from sklearn.metrics import log_loss
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

def nll(probs, labels):
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    return log_loss(labels, probs)

def accuracy(probs, labels):
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    probs = np.argmax(probs, 1)
    labels = np.argmax(labels, 1)
    return sum(labels == probs) * 1.0 / len(labels)


