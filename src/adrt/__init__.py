# Copyright (c) 2022 Karl Otness, Donsub Rim
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


__all__ = ["adrt", "iadrt", "bdrt", "iadrt_fmg", "utils", "core"]
__version__ = "0.1.0"


import itertools
import numpy as np
from ._wrappers import adrt, iadrt, bdrt
from . import utils, core


def iadrt_fmg(a, /, *, max_iters=None):
    if a.ndim > 3:
        raise ValueError(
            f"Batch dimension not supported for iadrt_fmg, got {a.ndim} dimensions"
        )
    if max_iters is not None and max_iters < 1:
        raise ValueError(f"Must allow at least one iteration (requested {max_iters})")
    # Following recipe for itertools.pairwise from Python 3.10
    i1, i2 = itertools.tee(
        map(
            # Pair each estimated inverse x with its residual error
            lambda x: (x, np.linalg.norm(adrt(x) - a)),
            # Use itertools.islice to limit iterations if requested
            itertools.islice(core.iadrt_fmg_iter(a, copy=False), max_iters),
        ),
        2,
    )
    next(i2, None)
    # Chain i2 with one extra value so we don't exhaust early
    # Use np.inf so the residual will certainly rise and we won't continue iterating
    for (_inv1, res1), (_inv2, res2) in zip(i1, itertools.chain(i2, [(None, np.inf)])):
        if res2 >= res1:
            # Residual failed to decrease, stop early
            break
    # Create a copy so returned array is writable (we have views from iadrt_fmg_iter)
    return _inv1.copy()
