import numpy as np
import torch


def calc_p(ncal, ngt, neq, smoothing=False):
    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)


def sort_sum(scores):
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


def coverage(preds, targets):
    covered = 0
    size = 0
    if torch.is_tensor(targets):
        targets = targets.squeeze().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    preds = np.asarray(preds)
    targets = np.expand_dims(targets.flatten(), axis=1)
    return np.mean(np.take_along_axis(preds, targets, axis=1))


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]
