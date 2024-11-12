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


import math
import pytest
import numpy as np
import adrt


def test_reject_negative():
    with pytest.raises((OverflowError, ValueError)):
        adrt.core.num_iters(-1)


@pytest.mark.parametrize(
    "in_val", [4.0, None, pytest.param((), id="tuple"), pytest.param([], id="list")]
)
def test_reject_non_integer(in_val):
    with pytest.raises(TypeError):
        adrt.core.num_iters(in_val)


def test_zero():
    assert adrt.core.num_iters(0) == 0


def test_one():
    assert adrt.core.num_iters(1) == 0


def test_bool():
    assert adrt.core.num_iters(False) == adrt.core.num_iters(0)
    assert adrt.core.num_iters(True) == adrt.core.num_iters(1)


@pytest.mark.parametrize("in_val", range(2, 17))
def test_common_value(in_val):
    target = math.ceil(math.log2(in_val))
    assert adrt.core.num_iters(in_val) == target


@pytest.mark.parametrize(
    "cls", [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
def test_numpy_integer(cls):
    value = 20
    assert adrt.core.num_iters(cls(value)) == adrt.core.num_iters(value)


@pytest.mark.parametrize("cls", [np.float32, np.float64, np.complex64, np.complex128])
def test_reject_numpy_non_integer(cls):
    with pytest.raises(TypeError):
        adrt.core.num_iters(cls(4))


@pytest.mark.parametrize("exp", [32, 64, 128])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_large_value(exp, offset):
    assert adrt.core.num_iters((2**exp) + offset) == exp + (1 if offset > 0 else 0)
