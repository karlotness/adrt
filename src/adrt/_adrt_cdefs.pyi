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

import numpy as np
import typing
import numpy.typing as npt

F = typing.TypeVar("F", np.float32, np.float64)

OPENMP_ENABLED: typing.Final[bool]

def adrt(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def adrt_step(a: npt.NDArray[F], step: int, /) -> npt.NDArray[F]: ...
def iadrt(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def bdrt(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def bdrt_step(a: npt.NDArray[F], step: int, /) -> npt.NDArray[F]: ...
def interp_to_cart(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def press_fmg_restriction(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def press_fmg_prolongation(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
def press_fmg_highpass(a: npt.NDArray[F], /) -> npt.NDArray[F]: ...
