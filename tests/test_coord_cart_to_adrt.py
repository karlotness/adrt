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
import itertools
import adrt


def test_return_value_order():
    ret = adrt.utils.coord_cart_to_adrt(
        theta=np.array([-np.pi / 4, 0, np.pi / 4]),
        t=np.array([-0.25, 0, 0.25]),
        n=8,
    )
    quadrant, height, slope, factor = ret
    assert quadrant is ret.quadrant
    assert height is ret.height
    assert slope is ret.slope
    assert factor is ret.factor


@pytest.mark.parametrize(
    "theta_dtype, t_dtype", itertools.product([np.float32, np.float64], repeat=2)
)
def test_return_dtype(theta_dtype, t_dtype):
    ret = adrt.utils.coord_cart_to_adrt(
        theta=np.array([-np.pi / 4, 0, np.pi / 4]).astype(theta_dtype),
        t=np.array([-0.25, 0, 0.25]).astype(t_dtype),
        n=8,
    )
    assert ret.quadrant.dtype == np.dtype(np.uint8)
    assert ret.height.dtype == np.dtype(np.int64)
    assert ret.slope.dtype == np.dtype(np.uint64)
    assert ret.factor.dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    "theta, quadrant",
    [
        pytest.param(-3 * np.pi / 8, 0, id="-3pi/8"),
        pytest.param(-1 * np.pi / 8, 1, id="-pi/8"),
        pytest.param(np.pi / 8, 2, id="pi/8"),
        pytest.param(3 * np.pi / 8, 3, id="3pi/8"),
    ],
)
def test_quadrant_midpoints(theta, quadrant):
    ret = adrt.utils.coord_cart_to_adrt(theta=np.array([theta]), t=np.zeros(1), n=8)
    assert ret.quadrant.item() == quadrant


@pytest.mark.parametrize("period", [-2, -1, 1, 2])
def test_quadrant_midpoints_periodic(period):
    theta = np.arange(-3, 4, 2) * np.pi / 8
    t = np.linspace(-1 / 16, 1 / 16, 4)
    n = 8
    ret_base = adrt.utils.coord_cart_to_adrt(theta=theta, t=t, n=n)
    ret = adrt.utils.coord_cart_to_adrt(theta=(theta + period * np.pi), t=t, n=n)
    assert np.all(ret_base.quadrant == ret.quadrant)
    assert np.all(ret_base.height == ret.height)
    assert np.all(ret_base.slope == ret.slope)
    assert np.allclose(ret_base.factor, ret.factor)


def test_refuses_mismatched_array_shapes():
    with pytest.raises(ValueError):
        _ = adrt.utils.coord_cart_to_adrt(
            theta=np.array([-np.pi / 4, 0, np.pi / 4]),
            t=np.array([-0.25, 0]),
            n=8,
        )


def test_reject_invalid_size():
    theta = np.array([0.25 * np.pi])
    t = np.array([0.5])
    n = 1
    with pytest.raises(ValueError):
        adrt.utils.coord_cart_to_adrt(theta, t, n)


size = 2 ** 2
theta0 = np.array([0.25 * np.pi - 1e-8])
theta1 = np.array([0.0 * np.pi + 1e-8])
theta2 = np.array([0.25 * np.pi + 1e-8])
theta3 = np.array([0.5 * np.pi - 1e-8])
theta4 = np.array([-0.25 * np.pi + 1e-8])
theta5 = np.array([0.0 * np.pi - 1e-8])
theta6 = np.array([-0.5 * np.pi + 1e-8])
theta7 = np.array([-0.25 * np.pi - 1e-8])
t0 = np.array([(1 - 1 / size) / np.sqrt(2)])
t1 = np.array([(1 / size - 1) / np.sqrt(2)])
t2 = np.array([(1 / size - 1) * 0.5])
t3 = np.array([(1 - 1 / size) * 0.5])


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((theta0, t0, size), (2, 2 * size - 2, size - 1)),
        ((theta0, t1, size), (2, 0, size - 1)),
        ((theta1, t2, size), (2, 0, 0)),
        ((theta1, t3, size), (2, size - 1, 0)),
        ((theta2, t0, size), (3, 0, size - 1)),
        ((theta2, t1, size), (3, 2 * size - 2, size - 1)),
        ((theta3, t2, size), (3, size - 1, 0)),
        ((theta3, t3, size), (3, 0, 0)),
        ((theta4, t3, size), (1, 1, size - 1)),
        ((theta4, t1, size), (1, 2 * size - 2, size - 1)),
        ((theta5, t1, size), (1, size, 0)),
        ((theta5, t3, size), (1, 0, 0)),
        ((theta7, t0, size), (0, 2 * size - 2, size - 1)),
        ((theta7, t1, size), (0, 0, size - 1)),
        ((theta6, t2, size), (0, 0, 0)),
        ((theta6, t3, size), (0, size - 1, 0)),
    ],
)
def test_corners(test_input, expected):

    theta = test_input[0]
    t = test_input[1]
    n = test_input[2]

    coord = adrt.utils.coord_cart_to_adrt(theta, t, n)

    quadrant_expected = expected[0]
    height_expected = expected[1]
    slope_expected = expected[2]

    if np.abs(theta) > 0.25 * np.pi:
        th0 = np.abs(theta) - 0.5 * np.pi
    else:
        th0 = theta

    assert np.all(coord.quadrant == quadrant_expected)
    assert np.all(coord.height == height_expected)
    assert np.all(coord.slope == slope_expected)
    assert np.isclose(coord.factor, 1 / np.cos(th0))
