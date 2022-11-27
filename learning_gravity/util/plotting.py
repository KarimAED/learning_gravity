from typing import NoReturn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_field(
    positions: np.ndarray, forces: np.ndarray, mass_distribution: np.ndarray
) -> NoReturn:
    plt.figure()
    plt.xlabel("x in A.U.")
    plt.ylabel("y in A.U.")
    abs_forces = np.expand_dims(np.sqrt(np.square(forces).sum(axis=1)), axis=1)
    plt.quiver(*positions.T, *forces.T, scale=np.max(abs_forces) * 10, label="Field")
    plt.scatter(
        mass_distribution[1],
        mass_distribution[2],
        s=mass_distribution[0] * 20,
        c="r",
        label="Masses",
    )
    plt.legend(loc="upper right")
    plt.tight_layout()


def plot_mass_history(
    positions: np.ndarray,
    forces: np.ndarray,
    mass_distribution: np.ndarray,
    mass_history: np.ndarray,
    pos_history: np.ndarray,
):
    plot_field(positions, forces, mass_distribution)
    num_frames = mass_history.shape[0]
    fig = plt.gcf()
    (scatter,) = plt.plot([], [], "bo")

    def init():
        return (scatter,)

    def update(frame: int, scatter_obj: object, pos_hist: np.ndarray):
        positions = pos_hist[frame].T
        scatter_obj.set_data(positions[0], positions[1])
        return (scatter_obj,)

    animation = FuncAnimation(
        fig,
        partial(update, scatter_obj=scatter, pos_hist=pos_history),
        frames=range(num_frames),
        init_func=init,
        interval=100,
    )
    return animation
