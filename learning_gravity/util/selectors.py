import torch

from learning_gravity.util.evaluation_metrics import masses, positions
from learning_gravity.transforms import Transforms

LOSSES = {"l1": torch.nn.L1Loss, "l2": torch.nn.MSELoss}

OPTIMIZERS = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}

METRICS = {"masses": masses, "positions": positions}


def loss_selector(loss_str: str):
    assert loss_str in LOSSES
    return LOSSES[loss_str]


def optimizer_selector(optimizer_str: str):
    assert optimizer_str in OPTIMIZERS
    return OPTIMIZERS[optimizer_str]


def metric_selector(metric_str: str):
    assert metric_str in METRICS
    return METRICS[metric_str]


def transform_selector(transform_str: str):
    assert transform_str in Transforms
    return Transforms[transform_str]
