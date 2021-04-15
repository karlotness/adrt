# Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
# All rights reserved
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
import ctypes
import pytest
import adrt


SIZE_T_BITS = ctypes.sizeof(ctypes.c_size_t) * 8
SIZE_T_MAX = (2 ** SIZE_T_BITS) - 1


def test_reject_negative():
    with pytest.raises(OverflowError):
        adrt.core.num_iters(-1)


def test_reject_overflow():
    with pytest.raises(OverflowError):
        adrt.core.num_iters(SIZE_T_MAX + 1)


@pytest.mark.parametrize("in_val", [4.0, None, (), []])
def test_reject_non_integer(in_val):
    with pytest.raises(TypeError):
        adrt.core.num_iters(in_val)


def test_size_t_max():
    assert adrt.core.num_iters(SIZE_T_MAX) == SIZE_T_BITS


def test_zero():
    assert adrt.core.num_iters(0) == 0


def test_one():
    assert adrt.core.num_iters(1) == 0


@pytest.mark.parametrize("in_val", range(2, 17))
def test_common_value(in_val):
    target = math.ceil(math.log2(in_val))
    assert adrt.core.num_iters(in_val) == target