from torch.nn import Module
from torch.utils.data import DataLoader


def masses(model: Module, training_loader: DataLoader, val_loader: DataLoader):
    return model.masses.tolist()


def positions(model: Module, training_loader: DataLoader, val_loader: DataLoader):
    return model.positions.tolist()
