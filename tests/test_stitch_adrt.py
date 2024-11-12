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
import adrt


def _check_column_contiguous(stitched):
    n = stitched.shape[-1] // 4
    assert np.allclose(
        stitched[..., :, n - 1], stitched[..., :, n]
    ), "First seam does not match"
    assert np.allclose(
        stitched[..., :, 2 * n - 1], stitched[..., :, 2 * n]
    ), "Second seam does not match"
    assert np.allclose(
        stitched[..., :, 3 * n - 1], stitched[..., :, 3 * n]
    ), "Third seam does not match"
    # Last column needs to be flipped
    assert np.allclose(
        stitched[..., :, -1], np.flip(stitched[..., :, 0], axis=-1)
    ), "Start and end are not flipped copies of one another"


def _check_quadrant_ordering(stitched, out):
    n = stitched.shape[-1] // 4
    assert np.allclose(
        stitched[..., : 2 * n - 1, :n], out[..., 0, :, :]
    ), "Quadrant 0 positioned out of order"
    assert np.allclose(
        stitched[..., : 2 * n - 1, n : 2 * n], out[..., 1, ::-1, ::-1]
    ), "Quadrant 1 positioned out of order"
    assert np.allclose(
        stitched[..., -2 * n + 1 :, 2 * n : 3 * n], out[..., 2, :, :]
    ), "Quadrant 2 positioned out of order"
    assert np.allclose(
        stitched[..., -2 * n + 1 :, 3 * n :], out[..., 3, ::-1, ::-1]
    ), "Quadrant 3 positioned out of order"


def _check_zero_stencil(stitched):
    n = stitched.shape[-1] // 4
    # Test rectangular blocks of zeros
    assert np.all(
        stitched[..., -n + 1 :, : 2 * n] == 0
    ), "Non-zero value in lower left rectangular block"
    assert np.all(
        stitched[..., : n - 1, 2 * n :] == 0
    ), "Non-zero value in upper right rectangular block"
    # Check triangles of zeros
    triu_a, triu_b = np.triu_indices(2 * n - 1, m=n, k=1)
    tril_a, tril_b = np.tril_indices(2 * n - 1, m=n, k=-n)
    assert np.all(
        stitched[..., : 2 * n - 1, :n][..., tril_a, tril_b] == 0
    ), "Non-zero value in band 0 triangle"
    assert np.all(
        stitched[..., : 2 * n - 1, n : 2 * n][..., triu_a, triu_b] == 0
    ), "Non-zero value in band 1 triangle"
    assert np.all(
        stitched[..., n - 1 :, 2 * n : 3 * n][..., tril_a, tril_b] == 0
    ), "Non-zero value in band 2 triangle"
    assert np.all(
        stitched[..., n - 1 :, 3 * n :][..., triu_a, triu_b] == 0
    ), "Non-zero value in band 3 triangle"


def test_accepts_adrt_output():
    n = 16
    inarr = np.arange(n**2).reshape((n, n)).astype("float32")
    out = adrt.adrt(inarr)
    stitched = adrt.utils.stitch_adrt(out)
    assert stitched.shape == (3 * n - 2, 4 * n)
    _check_column_contiguous(stitched)
    _check_quadrant_ordering(stitched, out)
    _check_zero_stencil(stitched)


def test_accepts_adrt_output_batched():
    n = 16
    inarr = np.arange(3 * (n**2)).reshape((3, n, n)).astype("float32")
    out = adrt.adrt(inarr)
    stitched = adrt.utils.stitch_adrt(out)
    assert stitched.shape == (3, 3 * n - 2, 4 * n)
    _check_column_contiguous(stitched)
    _check_quadrant_ordering(stitched, out)
    _check_zero_stencil(stitched)


def test_accepts_adrt_output_remove_repeated():
    n = 16
    inarr = np.arange(n**2).reshape((n, n)).astype("float32")
    out = adrt.adrt(inarr)
    stitched = adrt.utils.stitch_adrt(out, remove_repeated=True)
    assert stitched.shape == (3 * n - 2, 4 * n - 4)
    # Check deleting repeated columns
    stitch_repeat = adrt.utils.stitch_adrt(out, remove_repeated=False)
    stitch_repeat = np.delete(stitch_repeat, [i * n - 1 for i in range(1, 5)], axis=-1)
    assert stitched.shape == stitch_repeat.shape
    assert np.allclose(stitched, stitch_repeat)


def test_accepts_adrt_output_remove_repeated_batched():
    n = 16
    inarr = np.arange(3 * (n**2)).reshape((3, n, n)).astype("float32")
    out = adrt.adrt(inarr)
    stitched = adrt.utils.stitch_adrt(out, remove_repeated=True)
    assert stitched.shape == (3, 3 * n - 2, 4 * n - 4)
    # Check deleting repeated columns
    stitch_repeat = adrt.utils.stitch_adrt(out, remove_repeated=False)
    stitch_repeat = np.delete(stitch_repeat, [i * n - 1 for i in range(1, 5)], axis=-1)
    assert stitched.shape == stitch_repeat.shape
    assert np.allclose(stitched, stitch_repeat)


def test_accepts_adrt_output_multi_batched():
    n = 8
    inarr = np.arange(6 * (n**2)).reshape((2, 3, n, n)).astype("float32")
    out_1 = adrt.adrt(inarr[0])
    out_2 = adrt.adrt(inarr[1])
    out = np.stack([out_1, out_2])
    stitched = adrt.utils.stitch_adrt(out)
    assert stitched.shape == (2, 3, 3 * n - 2, 4 * n)
    _check_column_contiguous(stitched)
    _check_quadrant_ordering(stitched, out)
    _check_zero_stencil(stitched)


def test_accepts_adrt_output_remove_repeated_multi_batched():
    n = 8
    inarr = np.arange(6 * (n**2)).reshape((2, 3, n, n)).astype("float32")
    out_1 = adrt.adrt(inarr[0])
    out_2 = adrt.adrt(inarr[1])
    out = np.stack([out_1, out_2])
    stitched = adrt.utils.stitch_adrt(out, remove_repeated=True)
    assert stitched.shape == (2, 3, 3 * n - 2, 4 * n - 4)
    # Check deleting repeated columns
    stitch_repeat = adrt.utils.stitch_adrt(out, remove_repeated=False)
    stitch_repeat = np.delete(stitch_repeat, [i * n - 1 for i in range(1, 5)], axis=-1)
    assert stitched.shape == stitch_repeat.shape
    assert np.allclose(stitched, stitch_repeat)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_accepts_multiple_dtypes(dtype):
    n = 8
    inarr = np.ones((4, 2 * n - 1, n))
    stitched = adrt.utils.stitch_adrt(inarr.astype(dtype))
    assert stitched.shape == (3 * n - 2, 4 * n)


def test_small_matrix():
    n = 1
    inarr = np.arange(n**2).reshape((n, n)).astype("float32")
    out = adrt.adrt(inarr)
    stitched = adrt.utils.stitch_adrt(out)
    assert stitched.shape == (3 * n - 2, 4 * n)
    _check_column_contiguous(stitched)
    _check_quadrant_ordering(stitched, out)
    _check_zero_stencil(stitched)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 0, 0), id="zero_dims"),
        pytest.param((4, 4, 2), id="mismatched_sizes"),
        pytest.param((7, 4), id="too_few_dims"),
    ],
)
def test_rejects_invalid_sizes(shape):
    inarr = np.ones(shape).astype("float32")
    with pytest.raises(ValueError, match="unsuitable shape .*ADRT output"):
        _ = adrt.utils.stitch_adrt(inarr)
