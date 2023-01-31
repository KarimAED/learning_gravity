from pathlib import Path
from typing import Dict, Any
import os
import json
import argparse
import time
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torchviz

from learning_gravity.dataset import PositionDataset
from learning_gravity.model import Model
from learning_gravity.util.plotting import plot_field
from learning_gravity.util.selectors import (
    loss_selector,
    optimizer_selector,
    metric_selector,
)


REQUIRED_KEYS = ["data", "training", "model"]


def parse_definition() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="gravity_learner")
    parser.add_argument("definition_file", help="file defining the learning set-up")
    args = parser.parse_args()
    definition_path = Path("run_definitions") / args.definition_file
    assert (
        definition_path.exists() and definition_path.is_file()
    ), "definition_file must point to valid yaml file."
    try:
        with open(definition_path, "r", encoding="utf-8") as definition_file:
            definition_dict = yaml.safe_load(definition_file)
    except Exception as exc:
        raise ValueError(
            "definition_file does not point to a valid yaml file."
        ) from exc

    assert all(
        k in definition_dict for k in REQUIRED_KEYS
    ), f"yaml file must contain all of the following keys: {REQUIRED_KEYS}"
    definition_dict["name"] = args.definition_file.split(".")[0]
    return definition_dict


def load_data(data_definition: Dict[str, Any]) -> Dict[str, Any]:
    folder = Path("datasets") / data_definition["dataset"]
    assert (
        folder.is_dir()
    ), f"Dataset {data_definition['dataset']} must point to directory."
    assert all(
        folder / file in folder.iterdir()
        for file in ("masses.txt", "positions.txt", "forces.txt")
    )

    mass_positions = np.loadtxt(folder / "masses.txt", skiprows=1).T
    pos_ds = PositionDataset.from_files(folder)
    pos_ds.apply_transforms(
        data_definition["input_transforms"], data_definition["output_transforms"]
    )
    if data_definition["splits"] is not None:
        datasets = pos_ds.split(data_definition["splits"])
    else:
        datasets = [pos_ds]
    data_dict = {
        "mass_positions": mass_positions,
    }
    if len(datasets) == 1:
        data_dict["train_ds"] = datasets[0]
    elif len(datasets) == 2:
        data_dict["train_ds"] = datasets[0]
        data_dict["test_ds"] = datasets[-1]
    elif len(datasets) == 3:
        data_dict["train_ds"] = datasets[0]
        data_dict["val_ds"] = datasets[1]
        data_dict["test_ds"] = datasets[-1]
    else:
        raise NotImplementedError(
            "Splits into more than 3 subsets are not implemented."
        )
    return data_dict


def main():
    try:
        os.chdir("learning_gravity")
    except:
        raise Exception("Cannot navigate to proper directory.")
    definition_dict = parse_definition()
    time_stamp = int(time.time())
    out_path = Path("runs") / f"{time_stamp}_{definition_dict['name']}"
    assert not out_path.exists()
    os.mkdir(out_path)
    with open(out_path / "definition.json", "w", encoding="utf-8") as definition_log:
        json.dump(definition_dict, definition_log, indent=4)

    data_dict = load_data(definition_dict["data"])
    if definition_dict["plotting"]["inputs"]:
        plot_field(
            positions=data_dict["train_ds"].raw_positions,
            forces=data_dict["train_ds"].raw_forces,
            mass_distribution=data_dict["mass_positions"],
        )
        plt.savefig(out_path / "forces.pdf")

    training_definition = definition_dict["training"]
    shuffle = training_definition["shuffle"]
    batch_size = training_definition["batch_size"]

    train_loader = DataLoader(
        data_dict["train_ds"], shuffle=shuffle, batch_size=batch_size
    )
    validate = False
    test = False
    if "test_ds" in data_dict:
        test = True
        test_loader = DataLoader(
            data_dict["test_ds"], shuffle=shuffle, batch_size=batch_size
        )
    if "val_ds" in data_dict:
        validate = True
        val_loader = DataLoader(
            data_dict["val_ds"], shuffle=shuffle, batch_size=batch_size
        )

    model_definition = definition_dict["model"]
    if "args" not in model_definition:
        model_definition["args"] = []
    if "kwargs" not in model_definition:
        model_definition["kwargs"] = {}

    model = Model.factory(
        model_definition["variant"],
        *model_definition["args"],
        **model_definition["kwargs"],
    )

    optimizer = optimizer_selector(training_definition["optimizer"])(
        model.parameters(), lr=float(training_definition["lr"])
    )
    criterion = loss_selector(training_definition["loss"])(reduction="mean")

    log_definition = definition_dict["logging"]
    if "epoch_spacing" in log_definition:
        epoch_spacing = log_definition["epoch_spacing"]
    else:
        epoch_spacing = 1
    if "metrics" in log_definition:
        metrics = log_definition["metrics"]
    else:
        metrics = []

    if log_definition["history"]:
        log_stream = open(out_path / "history.txt", "w", encoding="utf-8")

    loss = 0
    for epoch in range(training_definition["epochs"]):
        loss = 0
        for batch in train_loader:
            label_pred = model(batch[0])
            loss += criterion(label_pred, batch[1]) / len(train_loader)

        if epoch % epoch_spacing == 0:
            log_data = {"epoch": epoch, "train_loss": loss.item()}
            if validate:
                val_loss = 0
                for batch in val_loader:
                    label_pred = model(batch[0])
                    val_loss += criterion(label_pred, batch[1]) / len(val_loader)
                log_data["val_loss"] = val_loss.item()
            for metric in metrics:
                metric_fn = metric_selector(metric)
                log_data[metric] = metric_fn(model, train_loader, val_loader)
            print(log_data)
            if log_definition["history"]:
                log_stream.write(str(log_data) + "\n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if log_definition["history"]:
        log_stream.close()

    predictions = None
    positions = None
    targets = None
    for batch in test_loader:
        if predictions is None:
            predictions = model(batch[0])
            positions = batch[0]
            targets = batch[1]
        else:
            predictions = torch.cat((predictions, model(batch[0])), dim=0).float()
            positions = torch.cat((positions, batch[0]), dim=0).float()
            targets = torch.cat((targets, batch[1]), dim=0).float()

    if definition_dict["plotting"]["predictions"]:
        plot_field(
            positions=data_dict["test_ds"].invert_feature(positions),
            forces=data_dict["test_ds"].invert_output(predictions),
            mass_distribution=data_dict["mass_positions"],
        )
        plt.savefig(out_path / "predictions.pdf")

    if definition_dict["plotting"]["difference"]:
        difference = data_dict["test_ds"].invert_output(predictions) - data_dict["test_ds"].invert_output(targets)
        plot_field(
            positions=data_dict["test_ds"].invert_feature(positions),
            forces=difference,
            mass_distribution=data_dict["mass_positions"],
        )
        plt.savefig(out_path / "difference.pdf")


if __name__ == "__main__":
    main()
