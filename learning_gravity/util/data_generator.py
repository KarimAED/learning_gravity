"""
Module containing the DataGenerator class, which is used to generate samples for arbitrary systems.
"""
from pathlib import Path
from typing import Iterable, NoReturn
import numpy as np
import torch

from learning_gravity.model import MassOptimizerModel


class DataGenerator:
    """
    Class to generate random data from a system definition file and save the data.
    """

    def __init__(
        self, directory_str: str, overwrite_not_append: bool = True
    ) -> NoReturn:
        """Initialize a data generator by pointing it to a relevant directory.

        Args:
            directory_str: str, name of the directory where the system definition can be found. Must
                be located in the project folder and must contain "masses.txt" file.
            overwrite_not_append: bool, whether to overwrite the previously collected data or append
                to it. Defaults to True, appending not yet implemented.
        """

        # make sure the directory exists and has definition file
        self._dir_path = Path("learning_gravity") / "datasets" / directory_str
        assert (
            self._dir_path.is_dir()
        ), "Directory string must point to a valid directory"
        assert (
            self._dir_path / "masses.txt" in self._dir_path.iterdir()
        ), "File defining system masses must be present"

        # allow to append data instead of overwriting in the future
        if not overwrite_not_append:
            raise NotImplementedError("Appending data is not yet supported")

        self._system_data = torch.tensor(
            np.loadtxt(self._dir_path / "masses.txt", skiprows=1)
        )
        # extract dimensionality of the system from the system definition
        self._dims = len(self._system_data[0][1:])

    def generate_random_samples(
        self,
        num_samples: int = 100,
        sample_range: Iterable[float] = (-10.0, 10.0),
        min_dist: float = 0.1,
    ) -> NoReturn:
        """Method to generate a set number of random samples within a position interval

        Args:
            num_samples: int, number of random samples to collect, defaults to 100.
            sample_range: Iterable[float], Iterable of 2 floats indicating the lower and upper
                bounds of the sampling space in the x and y directions, defaults to (-10., 10.).
            min_dist: float, minimum distance between sample points and system bodies, required
                to avoid singular behaviour near system bodies, defaults to 0.1.

        Returns: NoReturn

        """

        # weigh all samples equally by test mass 1
        test_mass = torch.tensor(1.0)
        forces = []
        positions = []

        # ensure correct number of samples is generated
        while len(positions) < num_samples:
            # generate random sample drawn from flat distribution within sample_range
            pos = sample_range[0] + (sample_range[1] - sample_range[0]) * torch.rand(
                self._dims
            )
            too_close = False
            force = torch.zeros_like(self._system_data[0, 1:])

            # find forces from all bodies in the system and sum
            for system_body in self._system_data:
                system_pos = system_body[1:]
                system_mass = system_body[0]

                # ensure that the near-field with singular behaviour is blocked out
                if (pos - system_pos).square().sum().sqrt() < min_dist:
                    too_close = True
                else:
                    force += MassOptimizerModel.analytical_force(
                        pos, system_pos, test_mass, system_mass
                    )

            # only include data if it is sufficiently far from all system bodies
            if not too_close:
                positions.append(pos)
                forces.append(force)

        # convert to torch tensors
        positions = torch.stack(positions)
        forces = torch.stack(forces)

        # save to txt files
        np.savetxt(self._dir_path / "positions.txt", positions.detach().numpy())
        np.savetxt(self._dir_path / "forces.txt", forces.detach().numpy())
