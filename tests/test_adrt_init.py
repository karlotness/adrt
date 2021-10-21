import pytest
import numpy as np
import adrt


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_single_all_ones(dtype):
    size = 16
    in_arr = np.ones((size, size)).astype(dtype)
    out_arr = adrt.core.adrt_init(in_arr)
    assert in_arr.dtype == out_arr.dtype
    assert out_arr.shape == (4, 2 * size - 1, size)
    assert np.all(out_arr[:, :size, :] == 1)
    assert np.all(out_arr[:, size:, :] == 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_single_unique_values(dtype):
    size = 16
    in_arr = np.arange(size ** 2).reshape(size, size).astype(dtype)
    out_arr = adrt.core.adrt_init(in_arr)
    values = set(in_arr.astype(np.int32).ravel())
    assert in_arr.dtype == out_arr.dtype
    assert out_arr.shape == (4, 2 * size - 1, size)
    assert set(out_arr.astype(np.int32).ravel()) == values
    assert np.all(out_arr[:, size:, :] == 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_batch_unique_values(dtype):
    size = 16
    batches = 3
    in_arr = np.arange(batches * size ** 2).reshape(batches, size, size).astype(dtype)
    batch_out_arr = adrt.core.adrt_init(in_arr)
    single_out_arr = np.stack([adrt.core.adrt_init(in_arr[i]) for i in range(batches)])
    assert batch_out_arr.shape[0] == batches
    assert batch_out_arr.ndim == 4
    assert batch_out_arr.shape == single_out_arr.shape
    assert batch_out_arr.dtype == single_out_arr.dtype
    assert np.allclose(batch_out_arr, single_out_arr)


def test_refuses_non_array():
    with pytest.raises(TypeError):
        adrt.core.adrt_init(None)
    with pytest.raises(TypeError):
        adrt.core.adrt_init(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )


def test_refuses_too_many_dims():
    in_arr = np.ones((2, 3, 16, 16)).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)


def test_refuses_too_few_dims():
    in_arr = np.ones(16).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)


def test_refuses_non_square():
    in_arr = np.ones((3, 16, 15)).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)
    in_arr = np.ones((15, 16)).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)


def test_refuses_non_power_of_two():
    in_arr = np.ones((7, 7)).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)
    in_arr = np.ones((2, 7, 7)).astype("float32")
    with pytest.raises(ValueError):
        adrt.core.adrt_init(in_arr)


def test_refuses_zero_axis_array():
    inarr = np.zeros((0, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        adrt.core.adrt_init(inarr)
