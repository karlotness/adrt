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


import pytest
import numpy as np
import adrt


def test_reject_invalid_size_small():
    n = 1
    with pytest.raises(ValueError):
        adrt.utils.coord_adrt_to_cart(n)


def test_reject_invalid_size_odd():
    n = 5
    with pytest.raises(ValueError):
        adrt.utils.coord_adrt_to_cart(n)


class TestADRTToCart:
    def test_correct_hcat_8x8(self):
        size = 8
        coord = adrt.utils.coord_adrt_to_cart(size)
        coord_hcat = adrt.utils.coord_adrt_to_cart_hcat(size)

        for i in range(4):
            flipped_index = (
                2 * ((i + 1) // 2) * size
                - (i % 2)
                + (-1) ** i * np.arange(size, dtype=int)
            )

            assert np.all(coord_hcat.angle[..., flipped_index] == coord.angle[i])
            assert np.all(coord_hcat.offset[..., flipped_index] == coord.offset[i])

    def test_spot_adrt_to_cart_2x2(self):
        size = 2**2
        angle, offset = adrt.utils.coord_adrt_to_cart(size)
        theta = np.arctan(np.linspace(0, 1, size))

        assert np.all(angle[0] == -np.pi / 2 + theta)
        assert np.all(angle[1] == -theta)
        assert np.all(angle[2] == theta)
        assert np.all(angle[3] == np.pi / 2 - theta)

        assert np.all(offset[::2, size - 1, 0] - offset[::2, size, 0] == 1 / size)
        assert offset[2, size - 1, 0] == -(size // 2 - 0.5) / size
        assert offset[2, 0, 0] == -offset[2, size - 1, 0]
        assert np.isclose(
            offset[1::2, 2 * size - 2, size - 1], (size // 2 - 0.5) / size * np.sqrt(2)
        ).all()
        assert np.isclose(
            offset[::2, 2 * size - 2, size - 1], -(size // 2 - 0.5) / size * np.sqrt(2)
        ).all()
        assert np.isclose(
            offset[::2, 0, size - 1], (size // 2 - 0.5) / size * np.sqrt(2)
        ).all()
        assert np.isclose(
            offset[1::2, 0, size - 1], -(size // 2 - 0.5) / size * np.sqrt(2)
        ).all()

    def test_spot_adrt_to_cart_hcat_angle_increasing(self):
        size=2**3
        angle, _ = adrt.utils.coord_adrt_to_cart_hcat(size, remove_repeated=True)
        angle = np.squeeze(angle)

        assert np.all(np.diff(angle) > 0)
