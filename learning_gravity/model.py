import abc
from pathlib import Path
from typing import Iterable, Callable, Optional, NoReturn, List, Tuple, Any
import numpy as np
import torch

from learning_gravity.util.selectors import transform_selector


class Model(torch.nn.Module):
    def __init__(
        self, preprocessing: List[str] = None, postprocessing: List[str] = None
    ) -> NoReturn:
        super().__init__()
        self._dims = None
        self._n_particles = None
        self._masses = None
        self._positions = None
        self._preprocessors = preprocessing
        self._postprocessors = postprocessing

    @property
    def positions(self) -> np.ndarray:
        return self._positions.detach().numpy()

    @property
    def masses(self) -> np.ndarray:
        return self._masses.detach().numpy()

    @staticmethod
    def factory(variant: str, *args, **kwargs):
        assert variant in VARIANTS
        return VARIANTS[variant](*args, **kwargs)

    def _preprocess_force(
        self,
        sample_pos: torch.Tensor,
        pos: torch.Tensor,
        mass_1: torch.Tensor,
        mass_2: torch.Tensor,
    ) -> Any:
        if self._preprocessors is None or not self._preprocessors:
            return sample_pos, pos, mass_1, mass_2

        for i, processor_str in enumerate(self._preprocessors):
            processor = transform_selector(processor_str)()
            if i == 0:
                processed_data = processor(sample_pos, pos, mass_1, mass_2)
            else:
                processed_data = processor(processed_data)
        return processed_data

    def _postprocess_force(self, force: torch.Tensor) -> torch.Tensor:
        if self._postprocessors is None or not self._postprocessors:
            return force

        for i, processor_str in enumerate(self._postprocessors):
            processor = transform_selector(processor_str)()
            if i == 0:
                processed_data = processor(force)
            else:
                processed_data = processor(processed_data)
        return processed_data

    def _processed_force(
        self,
        sample_pos: torch.Tensor,
        pos: torch.Tensor,
        mass_1: torch.Tensor,
        mass_2: torch.Tensor,
    ):
        force_inp = self._preprocess_force(sample_pos, pos, mass_1, mass_2)
        if isinstance(force_inp, tuple):
            force = self._force(*force_inp)
        else:
            force = self._force(force_inp)
        return self._postprocess_force(force)

    def forward(self, sample_pos: torch.Tensor):
        force = None
        for mass, pos in zip(self._masses, self._positions):
            if force is None:
                force = self._processed_force(
                    sample_pos, pos, torch.tensor([1.0]), mass.reshape(1)
                )
            else:
                force += self._processed_force(
                    sample_pos, pos, torch.tensor([1.0]), mass.reshape(1)
                )
        return force


class MassOptimizerModel(Model):
    def __init__(
        self,
        initial_masses: List[float],
        initial_positions: List[float],
        preprocessing: List[str] = None,
        postprocessing: List[str] = None,
    ) -> NoReturn:
        if not (preprocessing is None or not preprocessing):
            raise ValueError("Cannot use preprocessing with MassOptimizerModel.")
        super().__init__(postprocessing=postprocessing)
        self._force = self.analytical_force
        self._n_particles = len(initial_masses)
        self._dims = len(initial_positions[0])
        self._masses = torch.nn.Parameter(torch.tensor(initial_masses))
        self._positions = torch.nn.Parameter(torch.tensor(initial_positions))

    def _preprocess_force(
        self,
        sample_pos: torch.Tensor,
        pos: torch.Tensor,
        mass_1: torch.Tensor,
        mass_2: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        return sample_pos, pos, mass_1, mass_2

    @classmethod
    def analytical_force(
        cls,
        pos_1: torch.Tensor,
        pos_2: torch.Tensor,
        mass_1: torch.Tensor,
        mass_2: torch.Tensor,
    ) -> torch.Tensor:
        if len(pos_1.shape) > 1:
            dist = (pos_1 - pos_2).square().sum(axis=1).sqrt()
            dist = torch.reshape(dist, (len(dist), 1))
        else:
            dist = (pos_1 - pos_2).square().sum().sqrt()

        if np.any(np.isclose(dist.detach(), 0)):
            raise ValueError("Distance between two objects is 0")

        magnitude = 10.0 * mass_1 * mass_2 / dist.square()

        direction = (pos_1 - pos_2) / dist
        return -direction * magnitude


class ForceModel(Model):
    def __init__(
        self,
        system_folder: str,
        preprocessing: List[str] = None,
        postprocessing: List[str] = None,
    ):

        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
        system_path = Path("datasets") / system_folder
        assert system_path.is_dir(), "System folder does not point to a valid dir"
        assert (
            system_path / "masses.txt" in system_path.iterdir()
        ), "System folder does not contain system definition"
        system_data = torch.tensor(np.loadtxt(system_path / "masses.txt", skiprows=1)).T
        system_data = system_data.T[system_data[0] != 0].T
        self._dims = system_data.shape[0] - 1
        self._n_particles = system_data.shape[0]
        self._masses = system_data[0]
        self._positions = system_data[1:].T


class LinearForceModel(ForceModel):
    def __init__(
        self,
        system_folder: str,
        preprocessing: List[str] = None,
        postprocessing: List[str] = None,
    ):
        super().__init__(system_folder=system_folder, preprocessing=preprocessing, postprocessing=postprocessing)
        self._lin = torch.nn.Linear(self._dims + 2, self._dims + 1)

    def _force(self, inputs: torch.Tensor):
        return self._lin(inputs)


class NNForceModel(ForceModel):
    def __init__(
        self,
        system_folder: str,
        layer_list: List[int],
        input_dims: int,
        preprocessing: List[str] = None,
        postprocessing: List[str] = None,
    ):
        super().__init__(system_folder=system_folder, preprocessing=preprocessing, postprocessing=postprocessing)
        self._layers = [torch.nn.Linear(input_dims, layer_list[0])]
        for i, nodes in enumerate(layer_list[:-1]):
            layer = torch.nn.Linear(nodes, layer_list[i+1])
            self.__setattr__(f"layer_{i}", layer)
            self._layers.append(layer)

    def _force(
        self,
        inputs: torch.Tensor
    ):
        relu = torch.nn.ReLU()
        values = inputs
        for layer in self._layers[:-1]:
            values = layer(values)
            values = relu(values)
        return self._layers[-1](values)

VARIANTS = {
    "mass_optimizer": MassOptimizerModel,
    "linear_force": LinearForceModel,
    "nn_force": NNForceModel,
}
