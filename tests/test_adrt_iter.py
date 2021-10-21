import pytest
import numpy as np
import more_itertools as mi
import adrt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_match_adrt_all_ones(dtype):
    inarr = np.ones((16, 16)).astype(dtype)
    c_out = adrt.adrt(inarr)
    last = mi.last(adrt.core.adrt_iter(inarr))
    assert np.allclose(last, c_out)
    assert last.shape == c_out.shape
    assert last.dtype == c_out.dtype
    assert last.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_match_adrt_unique_values(dtype):
    size = 16
    inarr = np.arange(size ** 2).reshape((size, size)).astype(dtype)
    c_out = adrt.adrt(inarr)
    last = mi.last(adrt.core.adrt_iter(inarr))
    assert np.allclose(last, c_out)
    assert last.shape == c_out.shape
    assert last.dtype == c_out.dtype
    assert last.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_first_matches_adrt_init_batch(dtype):
    size = 16
    inarr = np.arange(3 * size ** 2).reshape((3, size, size)).astype(dtype)
    first = mi.first(adrt.core.adrt_iter(inarr))
    init = adrt.core.adrt_init(inarr)
    assert np.all(first == init)
    assert first.shape == init.shape
    assert first.dtype == init.dtype
    assert first.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_match_adrt_step_batch(dtype):
    size = 16
    inarr = np.arange(3 * size ** 2).reshape((3, size, size)).astype(dtype)
    for i, (a, b) in enumerate(mi.pairwise(adrt.core.adrt_iter(inarr))):
        step_out = adrt.core.adrt_step(a, step=i)
        assert np.allclose(b, step_out)
        assert b.shape == step_out.shape
        assert b.dtype == step_out.dtype
        assert b.dtype == np.dtype(dtype)


def test_refuses_int32():
    size = 16
    inarr = np.ones((size, size)).astype("int32")
    with pytest.raises(TypeError):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_non_square():
    size = 16
    inarr = np.ones((size, size - 1)).astype("float32")
    with pytest.raises(ValueError):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_non_power_of_two():
    size = 15
    inarr = np.ones((size, size)).astype("float32")
    with pytest.raises(ValueError):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_too_many_dim():
    size = 16
    inarr = np.ones((2, 3, size, size)).astype("float32")
    with pytest.raises(ValueError):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_too_few_dim():
    inarr = np.ones(16).astype("float32")
    with pytest.raises(ValueError):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_copy_default_returns_copy():
    size = 16
    inarr = np.ones((size, size)).astype("float32")
    assert all(a.flags.writeable and a.base is None for a in adrt.core.adrt_iter(inarr))


def test_copy_true_returns_copy():
    size = 16
    inarr = np.ones((size, size)).astype("float32")
    assert all(
        a.flags.writeable and a.base is None
        for a in adrt.core.adrt_iter(inarr, copy=True)
    )


def test_copy_false_returns_readonly():
    size = 16
    inarr = np.ones((size, size)).astype("float32")
    assert all(not a.flags.writeable for a in adrt.core.adrt_iter(inarr, copy=False))
