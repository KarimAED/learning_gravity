"""
Module defining the Dataset of positions and forces used to learn the system.
"""
from __future__ import annotations

from pathlib import Path
from typing import NoReturn, Dict, Optional, Callable, Union, Tuple

import numpy as np
from torch.utils.data import Dataset
import torch

from learning_gravity.transforms import Transforms


class PositionDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset that wraps data generated by the DataGenerator and read
    from file.
    """

    def __init__(
        self,
        positions: Union[np.ndarray, torch.Tensor],
        forces: Union[np.ndarray, torch.Tensor],
        dir_path: str = None,
    ) -> NoReturn:
        """Generic initializer used to create a position dataset from pre-read data.

        Args:
            positions: Union[np.ndarray, torch.Tensor], sampling positions of the force field.
            forces: Union[np.ndarray, torch.Tensor], force measured at the given positions.
            dir_path: str, optional, directory from which the data was loaded.
            normalize: bool, optional, whether to normalize the data or not.
        """
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions)
        if not isinstance(forces, torch.Tensor):
            forces = torch.tensor(forces)
        self._dir_path = dir_path
        self._positions = positions.clone()
        self._forces = forces.clone()
        self._input_transforms = []
        self._output_transforms = []

    def __len__(self) -> int:
        return self._positions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        return self._positions[idx, :], self._forces[idx, :]

    @property
    def positions(self) -> torch.Tensor:
        return self._positions.clone()

    @property
    def forces(self) -> torch.Tensor:
        return self._forces.clone()

    def invert_output(self, output: torch.Tensor) -> torch.Tensor:
        for transform in self._output_transforms[::-1]:
            output = transform.invert(output)
        return output.detach().numpy()

    def invert_feature(self, feature: torch.Tensor) -> torch.Tensor:
        for transform in self._input_transforms[::-1]:
            feature = transform.invert(feature)
        return feature.detach().numpy()

    @property
    def raw_positions(self) -> torch.Tensor:
        """

        Returns: Unscaled positions as a torch.Tensor array.

        """
        return self.invert_feature(self._positions.clone())

    @property
    def raw_forces(self) -> torch.Tensor:
        """

        Returns: Unscaled forces as a numpy array.

        """
        return self.invert_output(self._forces.clone())

    def split(self, ratio: Union[float, Tuple[float]] = 0.7) -> Tuple[PositionDataset]:
        """Method to split the data into two datasets.

        Args:
            ratio: float, fraction of number of samples to allocate into the first dataset.

        Returns:
            tuple of datasets, each represented by a tuple of torch tensors.
        """
        datasets = []
        if isinstance(ratio, float):
            ratio = [ratio, 1 - ratio]
        r_prev = 0
        assert np.isclose(sum(ratio), 1)
        assert all(0 <= rat <= 1 for rat in ratio)
        for rat in ratio:
            data_set = PositionDataset(
                *(
                    array.numpy()
                    for array in self[
                        int(r_prev * len(self)) : int((r_prev + rat) * len(self))
                    ]
                ),
                dir_path=self._dir_path,
            )
            data_set._input_transforms = self._input_transforms
            data_set._output_transforms = self._output_transforms
            datasets.append(data_set)
            r_prev = rat
        return datasets

    def apply_transforms(
        self,
        input_transforms: Optional[Union[str, list[str]]] = None,
        output_transforms: Optional[Union[str, list[str]]] = None,
    ) -> NoReturn:

        if input_transforms is not None:
            self.apply_input_transforms(input_transforms)
        if output_transforms is not None:
            self.apply_output_transforms(output_transforms)

    def apply_input_transform(self, transform_str: str) -> NoReturn:
        assert transform_str in Transforms
        transform = Transforms[transform_str]()
        self._positions = transform(self._positions)
        self._input_transforms.append(transform)

    def apply_output_transform(self, transform_str: str) -> NoReturn:
        assert transform_str in Transforms
        transform = Transforms[transform_str]()
        self._forces = transform(self._forces)
        self._output_transforms.append(transform)

    def apply_input_transforms(self, transforms: Union[str, list[str]]) -> NoReturn:
        if isinstance(transforms, str):
            self.apply_input_transform(transforms)
        else:
            for trans in transforms:
                self.apply_input_transform(trans)

    def apply_output_transforms(self, transforms: Union[str, list[str]]) -> NoReturn:
        if isinstance(transforms, str):
            self.apply_output_transform(transforms)
        else:
            for trans in transforms:
                self.apply_output_transform(trans)

    @classmethod
    def from_files(cls, data_directory: str) -> PositionDataset:
        """Initialize the Dataset from a given directory.

        Args:
            data_directory: str, name of the directory where the data is located.
            normalize: bool, optional, whether to normalize the data or not.
        """

        required_paths = ["masses.txt", "positions.txt", "forces.txt"]

        dir_path = Path(data_directory)

        assert dir_path.is_dir(), "Directory name has to point to a valid directory"

        for filename in required_paths:
            assert (
                dir_path / filename in dir_path.iterdir()
            ), "Files need to be present to load data"

        positions = torch.tensor(np.loadtxt(dir_path / "positions.txt"))
        forces = torch.tensor(np.loadtxt(dir_path / "forces.txt"))
        return cls(positions, forces, dir_path=dir_path)
