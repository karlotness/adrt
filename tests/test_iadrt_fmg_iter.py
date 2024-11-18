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


import itertools
import pytest
import numpy as np
import more_itertools as mi
import adrt


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_accepts_non_batch(dtype):
    size = 16
    inarr = (
        np.arange(4 * size * (2 * size - 1))
        .reshape((4, 2 * size - 1, size))
        .astype(dtype)
    )
    mi.consume(itertools.islice(adrt.core.iadrt_fmg_iter(inarr), 10))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_accepts_batch(dtype):
    size = 16
    inarr = (
        np.arange(3 * 4 * size * (2 * size - 1))
        .reshape((3, 4, 2 * size - 1, size))
        .astype(dtype)
    )
    mi.consume(itertools.islice(adrt.core.iadrt_fmg_iter(inarr), 10))


def test_refuses_too_many_dim():
    size = 16
    inarr = np.ones((2, 3, 4, 2 * size - 1, size)).astype("float32")
    with pytest.raises(ValueError, match="between 3 and 4 dimensions, but had 5"):
        mi.first(adrt.core.iadrt_fmg_iter(inarr), None)


def test_refuses_too_few_dim():
    size = 16
    inarr = np.ones((2 * size - 1, size)).astype("float32")
    with pytest.raises(ValueError, match="between 3 and 4 dimensions, but had 2"):
        mi.first(adrt.core.iadrt_fmg_iter(inarr), None)


def test_copy_default_returns_copy():
    size = 16
    inarr = np.zeros((4, 2 * size - 1, size), dtype="float32")
    assert all(
        a.flags.writeable and a.base is None
        for a in itertools.islice(adrt.core.iadrt_fmg_iter(inarr), 10)
    )


def test_copy_true_returns_copy():
    size = 16
    inarr = np.zeros((4, 2 * size - 1, size), dtype="float32")
    assert all(
        a.flags.writeable and a.base is None
        for a in itertools.islice(adrt.core.iadrt_fmg_iter(inarr, copy=True), 10)
    )


def test_copy_false_returns_readonly():
    size = 16
    inarr = np.zeros((4, 2 * size - 1, size), dtype="float32")
    assert all(
        not a.flags.writeable
        for a in itertools.islice(adrt.core.iadrt_fmg_iter(inarr, copy=False), 10)
    )


def test_small_few_steps_monotonic_error_decrease():
    size = 32
    steps = np.linspace(-10, 10, size)
    x, y = np.meshgrid(steps, steps)
    img = np.exp(np.negative(np.linalg.norm(np.stack((x, y), axis=-1), axis=-1)))
    residuals = np.linalg.norm(
        np.expand_dims(img, 0)
        - np.array(
            list(itertools.islice(adrt.core.iadrt_fmg_iter(adrt.adrt(img)), 10))
        ),
        axis=(-1, -2),
    )
    assert np.all(np.diff(residuals) <= 0)
