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
    inarr = np.arange(size**2).reshape((size, size)).astype(dtype)
    c_out = adrt.adrt(inarr)
    last = mi.last(adrt.core.adrt_iter(inarr))
    assert np.allclose(last, c_out)
    assert last.shape == c_out.shape
    assert last.dtype == c_out.dtype
    assert last.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_first_matches_adrt_init_batch(dtype):
    size = 16
    inarr = np.arange(3 * size**2).reshape((3, size, size)).astype(dtype)
    first = mi.first(adrt.core.adrt_iter(inarr))
    init = adrt.core.adrt_init(inarr)
    assert np.all(first == init)
    assert first.shape == init.shape
    assert first.dtype == init.dtype
    assert first.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_match_adrt_step_batch(dtype):
    size = 16
    inarr = np.arange(3 * size**2).reshape((3, size, size)).astype(dtype)
    for i, (a, b) in enumerate(mi.pairwise(adrt.core.adrt_iter(inarr))):
        step_out = adrt.core.adrt_step(a, step=i)
        assert np.allclose(b, step_out)
        assert b.shape == step_out.shape
        assert b.dtype == step_out.dtype
        assert b.dtype == np.dtype(dtype)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_correct_iter_length(size):
    inarr = np.ones((size, size)).astype("float32")
    num_elems = adrt.core.num_iters(inarr.shape[-1]) + 1
    assert mi.ilen(adrt.core.adrt_iter(inarr)) == num_elems


def test_refuses_int32():
    size = 16
    inarr = np.ones((size, size)).astype("int32")
    with pytest.raises(TypeError, match="int32"):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_non_square():
    size = 16
    inarr = np.ones((size, size - 1)).astype("float32")
    with pytest.raises(ValueError, match="must be square"):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_non_power_of_two():
    size = 15
    inarr = np.ones((size, size)).astype("float32")
    with pytest.raises(ValueError, match="power of two shape"):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_too_many_dim():
    size = 16
    inarr = np.ones((2, 3, size, size)).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 4"):
        mi.consume(adrt.core.adrt_iter(inarr))


def test_refuses_too_few_dim():
    inarr = np.ones(16).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 1"):
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
