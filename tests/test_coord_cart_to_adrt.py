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


def test_reject_theta_out_of_bounds():
    theta = np.array([np.pi])
    t = np.array([0.5])
    n = 2**2
    with pytest.raises(ValueError):
        adrt.utils.coord_cart_to_adrt(theta, t, n)


def test_reject_invalid_size():
    theta = np.array([0.25 * np.pi])
    t = np.array([0.5])
    n = 1
    with pytest.raises(ValueError):
        adrt.utils.coord_cart_to_adrt(theta, t, n)


def test_corners_4x4():
    theta = np.array([0.25 * np.pi - 1e-8])
    n = 2**4
    t = np.array([(1 / n - 1) / np.sqrt(2)])
    coord = adrt.utils.coord_cart_to_adrt(theta, t, n)

    assert np.all(coord.quadrant == 1)
    assert np.all(coord.height == 2 * n - 2)
    assert np.all(coord.slope == n - 1)
    assert np.isclose(coord.factor, 1 / np.cos(theta))

    theta = np.array([0.25 * np.pi - 1e-8])
    n = 2**4
    t = np.array([(1 - 1 / n) / np.sqrt(2)])
    coord = adrt.utils.coord_cart_to_adrt(theta, t, n)

    assert np.all(coord.quadrant == 1)
    assert np.all(coord.height == 0)
    assert np.all(coord.slope == n - 1)
    assert np.isclose(coord.factor, 1 / np.cos(theta))

    theta = np.array([0.0 * np.pi + 1e-8])
    n = 2**4
    t = np.array([(1 - 1 / n) * 0.5])
    coord = adrt.utils.coord_cart_to_adrt(theta, t, n)

    assert np.all(coord.quadrant == 1)
    assert np.all(coord.height == 0)
    assert np.all(coord.slope == 0)
    assert np.isclose(coord.factor, 1 / np.cos(theta))

    theta = np.array([0.0 * np.pi + 1e-8])
    n = 2**4
    t = np.array([(1 / n - 1) * 0.5])
    coord = adrt.utils.coord_cart_to_adrt(theta, t, n)

    assert np.all(coord.quadrant == 1)
    assert np.all(coord.height == n - 1)
    assert np.all(coord.slope == 0)
    assert np.isclose(coord.factor, 1 / np.cos(theta))
