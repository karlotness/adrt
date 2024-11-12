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


def test_return_value_order():
    ret = adrt.utils.coord_adrt(4)
    offset, angle = ret
    assert offset is ret.offset
    assert angle is ret.angle


def test_reject_invalid_size_small():
    with pytest.raises(
        ValueError, match="invalid Radon domain size 1, must be at least 2"
    ):
        adrt.utils.coord_adrt(1)


def test_reject_invalid_size_odd():
    with pytest.raises(
        ValueError, match="invalid Radon domain size 5, must be a power of two"
    ):
        adrt.utils.coord_adrt(5)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_return_correct_shape_dtype(n):
    ret = adrt.utils.coord_adrt(n)
    assert ret.offset.shape == (4, 2 * n - 1, n)
    assert ret.angle.shape == (4, 1, n)
    assert ret.offset.dtype == np.dtype(np.float64)
    assert ret.angle.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_angle_spacing(n):
    angle = adrt.utils.coord_adrt(n).angle.squeeze(1)
    # Check that angles are in [-pi/2, pi/2]
    assert np.all(angle >= -np.pi / 2)
    assert np.all(angle <= np.pi / 2)
    # Check that angles are linearly spaced by tan in the range [-pi/4, pi/4]
    angle_offsets = np.expand_dims(np.array([np.pi / 2, 0, 0, -np.pi / 2]), -1)
    expected_diffs = np.expand_dims(1 / (n - 1) * np.array([1, -1, 1, -1]), -1)
    angle_spaces = np.diff(np.tan(angle + angle_offsets))
    assert np.allclose(angle_spaces, expected_diffs)
    # Check first and last angle values
    expected_stops = np.array(
        [
            [-np.pi / 2, -np.pi / 4],
            [0, -np.pi / 4],
            [0, np.pi / 4],
            [np.pi / 2, np.pi / 4],
        ]
    )
    assert np.allclose(angle[:, [0, -1]], expected_stops)


def test_unique_stitches_angles_correctly():
    angle = adrt.utils.coord_adrt(16).angle
    unique = np.unique(angle)
    manually_stitched = np.concatenate(
        [
            angle[0, 0],
            np.flip(angle[1, 0, :-1]),
            angle[2, 0, 1:],
            np.flip(angle[3, 0, :-1]),
        ]
    )
    assert unique.shape == manually_stitched.shape
    assert np.allclose(unique, manually_stitched)


def test_sort_stitches_angles_correctly():
    angle = adrt.utils.coord_adrt(16).angle
    sorted_angles = np.sort(angle.ravel())
    manually_stitched = np.concatenate(
        [
            angle[0, 0],
            np.flip(angle[1, 0]),
            angle[2, 0],
            np.flip(angle[3, 0]),
        ]
    )
    assert sorted_angles.shape == manually_stitched.shape
    assert np.allclose(sorted_angles, manually_stitched)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_spot_check_offset(n):
    offset = adrt.utils.coord_adrt(n).offset
    # Check array repeats as tiles
    assert np.all(offset[:2] == offset[2:])
    # Check signs are opposite between first quadrants
    assert np.all(offset[0] == -offset[1])
    # Check columns are all linearly spaced
    col_differences = np.diff(offset[0], axis=0)
    assert np.allclose(col_differences, np.expand_dims(col_differences[0], 0))
    # Check spacing in the first column
    assert np.isclose(col_differences[0, 0], -1 / n)
    assert np.isclose(offset[0, n - 1, 0], (-n + 1) / (2 * n))
    assert np.isclose(offset[0, n - 1, 0], -offset[0, 0, 0])
    # Check values in the last column
    last_val = np.sqrt(2) * (n - 1) / (2 * n)
    assert np.isclose(offset[0, 0, -1], last_val)
    assert np.isclose(offset[0, -1, -1], -last_val)
