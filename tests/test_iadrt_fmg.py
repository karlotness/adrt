# Copyright 2023 Karl Otness, Donsub Rim
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import functools
import contextlib
import pytest
import numpy as np
import adrt


@pytest.fixture
def counting_iadrt_fmg_step(monkeypatch):
    count = 0
    orig_fn = adrt.core.iadrt_fmg_step

    @functools.wraps(orig_fn)
    def counting_inv(*args, **kwargs):
        nonlocal count
        count += 1
        return orig_fn(*args, **kwargs)

    monkeypatch.setattr(adrt.core, "iadrt_fmg_step", counting_inv)
    return lambda: count


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values(dtype):
    size = 8
    orig = np.arange(size**2).reshape((size, size)).astype(dtype)
    inarr = adrt.adrt(orig)
    inv = adrt.iadrt_fmg(inarr, max_iters=50)
    assert inv.dtype == orig.dtype
    assert inv.shape == orig.shape
    assert inv.flags.writeable
    assert np.allclose(inv, orig, atol=1e-3)


def test_rejects_non_integer_max_iters():
    size = 8
    inarr = np.zeros((4, 2 * size - 1, size), dtype="float32")
    with pytest.raises(ValueError, match="must be.*int"):
        _ = adrt.iadrt_fmg(inarr, max_iters=2.0)


def test_rejects_zero_max_iters():
    size = 8
    inarr = np.zeros((4, 2 * size - 1, size), dtype="float32")
    with pytest.raises(
        ValueError, match="must allow at least one iteration.*specified 0"
    ):
        _ = adrt.iadrt_fmg(inarr, max_iters=0)


def test_rejects_batch_dimension():
    size = 8
    inarr = np.zeros((3, 4, 2 * size - 1, size), dtype="float32")
    with pytest.raises(
        ValueError, match="batch dimension not supported.*got 4 dimensions"
    ):
        _ = adrt.iadrt_fmg(inarr, max_iters=50)


@pytest.mark.parametrize("val", [0, np.nan, np.inf, -np.inf])
def test_stops_quickly_on_edge_case(counting_iadrt_fmg_step, val):
    size = 8
    inarr = np.full((4, 2 * size - 1, size), val, dtype=np.float32)
    with np.errstate(invalid="ignore") if np.isinf(val) else contextlib.nullcontext():
        inv = adrt.iadrt_fmg(inarr, max_iters=50)
    count = counting_iadrt_fmg_step()
    assert count <= 2
    assert inv.shape == (size, size)
    assert inv.dtype == inarr.dtype
    assert np.isinf(val) or np.allclose(inv, val, equal_nan=True)


@pytest.mark.parametrize("max_iters", [1, 2, 3, 4])
def test_max_iters_limits_iterations(counting_iadrt_fmg_step, max_iters):
    size = 16
    orig = np.arange(size**2).reshape((size, size)).astype("float64")
    inarr = adrt.adrt(orig)
    _ = adrt.iadrt_fmg(inarr, max_iters=max_iters)
    count = counting_iadrt_fmg_step()
    assert count == max_iters
