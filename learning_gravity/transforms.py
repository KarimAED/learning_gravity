from typing import Optional, Callable, Any, Union
import abc
import numpy as np
import torch


class Transform:
    def __init__(self):
        pass

    def _get_dim(self, data: np.ndarray) -> int:
        if len(data.shape) in (1, 2):
            return len(data.shape)
        raise NotImplementedError(
            "Transform is only implemented for <=2 dimensional arrays."
        )

    @abc.abstractmethod
    def __call__(self, *args) -> Any:
        pass

    @abc.abstractmethod
    def invert(self, *args) -> Any:
        pass


class StandardizeTransform(Transform):
    def __init__(self):
        self._maxima = None

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self._get_dim(data)
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)

        self._maxima = torch.max(data, dim=0).values

        return data / self._maxima

    def invert(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        return data * self._maxima


class NormalizeTransform(Transform):
    def __init__(self):
        self._means = None
        self._stds = None

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self._get_dim(data)
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        self._means = torch.mean(data, keepdim=True, dim=0)
        temp_data = data - self._means
        self._stds = torch.std(temp_data, keepdim=True, dim=0)

        return temp_data / self._stds

    def invert(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        return (data * self._stds) + self._means


class LogMagnitudeTransform(Transform):
    def __init__(self):
        self._allowed_dims = [1, 2]

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        dim = self._get_dim(data)
        if dim == 1:
            data = data.reshape((len(data), 1))

        assert (
            data.shape[1] in self._allowed_dims
        ), f"Only implemented for {self._allowed_dims} number of values along second axis"

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        magnitudes = data.square().sum(dim=1).sqrt().reshape((data.shape[0], 1))
        log_magnitudes = magnitudes.log()
        temp_data = data / magnitudes

        return torch.cat((temp_data, log_magnitudes), dim=1).float()

    def invert(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:

        assert self._get_dim(data) == 2, "Input must be 2 dimensional"
        assert (
            data.shape[1] - 1 in self._allowed_dims
        ), f"Only implemented for {self._allowed_dims} number of values along second axis"
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        direction_data = data[:, :-1]
        log_magnitude = data[:, -1].reshape((data.shape[0], 1))
        magnitude = log_magnitude.exp()
        out_data = direction_data * magnitude
        if direction_data.shape[1] == 1:
            out_data = out_data.reshape((len(out_data)))
        return out_data


class LogDifferenceTransform(Transform):
    def __init__(self):
        self._offsets = None
        self._mass_1s = None
        self._log_magnitude_transform = LogMagnitudeTransform()
        self._required_width = 6
        self._allowed_dim = 2

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        assert self._get_dim(data) == self._allowed_dim
        assert data.shape[1] == self._required_width

        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)

        pos_1 = data[:, :2]
        pos_2 = data[:, 2:4]
        self._mass_1s = data[:, 4].reshape((data.shape[0], 1))
        mass_2 = data[:, 5].reshape((data.shape[0], 1))

        pos_diff = pos_2 - pos_1
        self._offsets = pos_1

        log_magnitude_diff = self._log_magnitude_transform(pos_diff)

        return torch.cat((log_magnitude_diff, torch.log(mass_2)), dim=1).float()

    def invert(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        assert self._get_dim(data) == self._allowed_dim
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        log_magnitude_diff = data[:, :-1]
        mass_2 = data[:, -1].reshape((data.shape[0], 1))
        pos_diff = self._log_magnitude_transform.invert(log_magnitude_diff)
        pos_2 = pos_diff + self._offsets

        return torch.cat((self._offsets, pos_2, self._mass_1s, torch.exp(mass_2)), dim=1).float()


class FlattenForceArgsTransform(Transform):
    def __call__(
        self,
        pos_1: Union[torch.Tensor, np.ndarray],
        pos_2: Union[torch.Tensor, np.ndarray],
        mass_1: Union[torch.Tensor, np.ndarray, float],
        mass_2: Union[torch.Tensor, np.ndarray, float],
    ) -> torch.Tensor:
        assert len(pos_1.shape) == 2
        assert len(pos_2.shape) == 1
        assert pos_1.shape[1] == len(pos_2)
        if not isinstance(pos_1, torch.Tensor):
            pos_1 = torch.tensor(pos_1)
        if not isinstance(pos_2, torch.Tensor):
            pos_2 = torch.tensor(pos_2)
        if not isinstance(mass_1, torch.Tensor):
            mass_1 = torch.tensor(mass_1)
        if not isinstance(mass_2, torch.Tensor):
            mass_2 = torch.tensor(mass_2)
        count = pos_1.shape[0]
        pos_2_cast = pos_2.unsqueeze(0).repeat(count, 1)
        mass_1_cast = mass_1.unsqueeze(0).repeat(count, 1)
        mass_2_cast = mass_2.unsqueeze(0).repeat(count, 1)
        return torch.cat((pos_1, pos_2_cast, mass_1_cast, mass_2_cast), dim=1).float()

    def invert(self, *args):
        raise NotImplementedError(
            "Inversion of flattening transform is not implemented."
        )


InputOutputTransforms = {
    "standardize": StandardizeTransform,
    "normalize": NormalizeTransform,
    "log_magnitude": LogMagnitudeTransform,
}

ForceTransforms = {
    "log_difference": LogDifferenceTransform,
    "flatten_force_args": FlattenForceArgsTransform,
}

Transforms = dict(**InputOutputTransforms, **ForceTransforms)
