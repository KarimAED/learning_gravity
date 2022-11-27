import pytest
import torch

from learning_gravity.dataset import PositionDataset
from learning_gravity.transforms import Transforms


class TestPositionDataset:
    def test_basic_dataset(self):
        positions = torch.rand((10_000, 2))
        forces = torch.rand((10_000, 2))
        dataset = PositionDataset(positions, forces)

        assert len(dataset) == 10_000
        pos_10, force_10 = dataset[10]
        assert torch.allclose(pos_10, positions[10, :])
        assert torch.allclose(force_10, forces[10, :])
        assert torch.allclose(dataset.raw_positions, positions)
        assert torch.allclose(dataset.raw_forces, forces)

    @pytest.mark.parametrize("splits", [(0.7, 0.3), (0.7, 0.15, 0.15), 0.7])
    def test_dataset_good_splits(self, splits):
        positions = torch.rand((10_000, 2))
        forces = torch.rand((10_000, 2))
        dataset = PositionDataset(positions, forces)

        split_datasets = dataset.split(splits)
        if isinstance(splits, tuple):
            assert len(split_datasets) == len(splits)
            for fraction, split in zip(splits, split_datasets):
                assert len(split) == int(fraction * len(dataset))
        else:
            assert len(split_datasets) == 2
            assert len(split_datasets[0]) == int(splits * len(dataset))

        assert sum(len(split) for split in split_datasets) == len(dataset)

    @pytest.mark.parametrize("splits", [(0.7, 0.3, 0.1), (-0.1, 1.1), 1.1, -0.1])
    def test_dataset_bad_splits(self, splits):
        positions = torch.rand((10_000, 2))
        forces = torch.rand((10_000, 2))
        dataset = PositionDataset(positions, forces)
        with pytest.raises(AssertionError):
            _ = dataset.split(splits)

    @pytest.mark.parametrize(
        "transform",
        ["normalize", "standardize", "log_magnitude", ["log_magnitude", "normalize"]],
    )
    def test_input_transform(self, transform):
        positions = torch.rand((10_000, 2))
        forces = torch.rand((10_000, 2))
        dataset = PositionDataset(positions, forces)
        dataset.apply_transforms(input_transforms=transform)
        if not isinstance(transform, list):
            transform = [transform]
        trans_positions = positions.clone()
        for trans in transform:
            transform_obj = Transforms[trans]()
            trans_positions = transform_obj(trans_positions)
        assert torch.allclose(dataset.positions, trans_positions)
        assert torch.allclose(dataset.raw_positions, positions)
