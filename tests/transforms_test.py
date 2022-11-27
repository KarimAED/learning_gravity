from typing import Tuple, Callable
import pytest
import torch

from learning_gravity.transforms import InputOutputTransforms, Transforms

IO_TRANSFORM_LIST = [value for key, value in InputOutputTransforms.items()]


@pytest.mark.parametrize("transform", IO_TRANSFORM_LIST)
class TestInputOutputTransforms:
    def test_consistency_1d(self, transform: Callable):
        test_data = torch.rand(1_000)
        trans = transform()
        transformed_data = trans(test_data)
        reconstructed_data = trans.invert(transformed_data)
        assert torch.allclose(reconstructed_data, test_data)

    def test_consistency_2d(self, transform: Callable):
        test_data = torch.rand((1_000, 2))
        trans = transform()
        transformed_data = trans(test_data)
        reconstructed_data = trans.invert(transformed_data)

        assert torch.allclose(reconstructed_data, test_data)

    def test_fails_with_3d(self, transform: Callable):
        with pytest.raises(NotImplementedError):
            test_data = torch.rand((1_000, 10, 100))
            trans = transform()
            transformed_data = trans(test_data)
            _ = trans.invert(transformed_data)


class TestLogDifferenceTransform:
    @pytest.mark.parametrize(
        "data_shape,error",
        [
            ((1_000,), AssertionError),
            ((1_000, 3), AssertionError),
            ((1_000, 6, 1), NotImplementedError),
        ],
    )
    def test_with_wrong_data_shape(self, data_shape: Tuple[int], error: Exception):
        test_data = torch.rand(data_shape)
        trans = Transforms["log_difference"]()
        with pytest.raises(error):
            trans(test_data)


class TestFlattenForceArgsTransform:
    @pytest.mark.parametrize(
        "pos_1_shape,pos_2_shape,mass_1,mass_2", [((1_000, 2), (2,), 1.0, 3.0)]
    )
    def test_with_good_data(
        self,
        pos_1_shape: Tuple[int],
        pos_2_shape: Tuple[int],
        mass_1: float,
        mass_2: float,
    ):
        pos_1 = torch.rand(pos_1_shape)
        pos_2 = torch.rand(pos_2_shape)
        trans = Transforms["flatten_force_args"]()
        transformed_data = trans(pos_1, pos_2, mass_1, mass_2)
        assert isinstance(transformed_data, torch.Tensor)
        assert transformed_data.shape[0] == pos_1_shape[0]
        assert transformed_data.shape[1] == pos_1_shape[1] * 2 + 2
        for i in range(pos_1_shape[0], 5):
            assert transformed_data[i, 2:4] == pos_2
            assert transformed_data[i, 4] == mass_1
            assert transformed_data[i, 5] == mass_2

    def test_inverse_not_implemented(self):
        trans = Transforms["flatten_force_args"]()
        with pytest.raises(NotImplementedError):
            trans.invert()

    @pytest.mark.parametrize(
        "pos_1_shape,pos_2_shape,mass_1,mass_2",
        [
            ((1_000), (2,), 1.0, 3.0),
            ((1_000, 2), (2, 1_000), 1.0, 3.0),
            ((1_000, 2), (3,), 1.0, 3.0),
        ],
    )
    def test_with_bad_data(
        self,
        pos_1_shape: Tuple[int],
        pos_2_shape: Tuple[int],
        mass_1: float,
        mass_2: float,
    ):
        pos_1 = torch.rand(pos_1_shape)
        pos_2 = torch.rand(pos_2_shape)
        trans = Transforms["flatten_force_args"]()
        with pytest.raises(AssertionError):
            _ = trans(pos_1, pos_2, mass_1, mass_2)
