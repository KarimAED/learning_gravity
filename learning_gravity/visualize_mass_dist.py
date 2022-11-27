from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from learning_gravity.util.history_loader import load_history
from learning_gravity.util.plotting import plot_mass_history


def visualize(run_dir: str, dataset_dir: str):
    history = load_history(run_dir)
    dataset_path = Path("learning_gravity") / "datasets" / dataset_dir
    mass_distribution = np.loadtxt(dataset_path / "masses.txt", skiprows=1).T
    pos_data = np.loadtxt(dataset_path / "positions.txt")
    force_data = np.loadtxt(dataset_path / "forces.txt")
    anim = plot_mass_history(
        positions=pos_data,
        forces=force_data,
        mass_distribution=mass_distribution,
        pos_history=history["positions"],
        mass_history=history["masses"],
    )
    anim_path = Path("learning_gravity") / "runs" / run_dir / "animation.gif"
    anim.save(anim_path, writer="Pillow")


if __name__ == "__main__":
    runs_dir = Path("learning_gravity") / "runs"
    for sub_folder in runs_dir.iterdir():
        print(f"visualizing run {sub_folder}")
        with open(sub_folder / "definition.json", "r", encoding="utf-8") as config:
            definition = json.load(config)
        ds = definition["data"]["dataset"]
        if (
            not (sub_folder / "animation.gif").exists()
            and definition["model"]["variant"] == "mass_optimizer"
        ):
            visualize(str(sub_folder).rsplit("/", maxsplit=1)[-1], ds)
        else:
            print("No visualization required.")
