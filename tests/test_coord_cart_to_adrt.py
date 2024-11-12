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
    with pytest.raises(ValueError, match="mismatched shapes for theta and t"):
        _ = adrt.utils.coord_cart_to_adrt(
            theta=np.zeros(3),
            t=np.zeros(2),
            n=8,
        )


def test_reject_invalid_size():
    theta = np.array([0.25 * np.pi])
    t = np.array([0.5])
    n = 1
    with pytest.raises(
        ValueError, match="invalid Radon domain size 1, must be at least 2"
    ):
        adrt.utils.coord_cart_to_adrt(theta, t, n)


def test_adrt_core_quadrants():
    n = 8
    adrt_coord = adrt.utils.coord_adrt(n)
    indices = adrt.utils.coord_cart_to_adrt(
        theta=np.broadcast_to(adrt_coord.angle, adrt_coord.offset.shape),
        t=adrt_coord.offset,
        n=n,
    )
    # Check quadrants
    assert np.all(
        indices.quadrant[:, :, 1:-1]
        == np.expand_dims(np.array([0, 1, 2, 3]), axis=(-1, -2))
    )
    assert np.all(
        np.logical_or(
            indices.quadrant[:2, :, -1] == 0,
            indices.quadrant[:2, :, -1] == 1,
        )
    )
    assert np.all(
        np.logical_or(
            indices.quadrant[-2:, :, -1] == 2,
            indices.quadrant[-2:, :, -1] == 3,
        )
    )
    assert np.all(indices.quadrant[0, :, 0] == 0)
    assert np.all(indices.quadrant[-1, :, 0] == 3)


def test_adrt_slope():
    n = 8
    adrt_coord = adrt.utils.coord_adrt(n)
    indices = adrt.utils.coord_cart_to_adrt(
        theta=np.broadcast_to(adrt_coord.angle, adrt_coord.offset.shape),
        t=adrt_coord.offset,
        n=n,
    )
    assert np.all(indices.slope == np.expand_dims(np.arange(n), (0, 1)))


def test_coords_adrt_identity():
    n = 8
    adrt_in = np.arange(n**2).reshape((n, n)).astype(np.float32)
    adrt_out = adrt.adrt(adrt_in)
    adrt_coord = adrt.utils.coord_adrt(n)
    indices = adrt.utils.coord_cart_to_adrt(
        theta=np.broadcast_to(adrt_coord.angle, adrt_coord.offset.shape),
        t=adrt_coord.offset,
        n=n,
    )
    adrt_indexed = adrt_out[
        indices.quadrant,
        indices.height,
        indices.slope,
    ]
    assert np.allclose(adrt_indexed, adrt_out)


size = 2**2
theta_0 = np.array([-0.50 * np.pi + 1e-8])
theta_1 = np.array([-0.25 * np.pi - 1e-8])
theta_2 = np.array([-0.25 * np.pi + 1e-8])
theta_3 = np.array([0.00 * np.pi - 1e-8])
theta_4 = np.array([0.00 * np.pi + 1e-8])
theta_5 = np.array([0.25 * np.pi - 1e-8])
theta_6 = np.array([0.25 * np.pi + 1e-8])
theta_7 = np.array([0.50 * np.pi - 1e-8])

t_0 = np.array([(1 - 1 / size) * 0.5])
t_1 = np.array([(1 / size - 1) * 0.5])
t_2 = np.array([(1 / size - 1) / np.sqrt(2)])
t_3 = np.array([(1 - 1 / size) / np.sqrt(2)])


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((theta_0, t_0, size), (0, 0, 0)),
        ((theta_0, t_1, size), (0, size - 1, 0)),
        ((theta_1, t_2, size), (0, 2 * size - 2, size - 1)),
        ((theta_1, t_3, size), (0, 0, size - 1)),
        ((theta_2, t_3, size), (1, 2 * size - 2, size - 1)),
        ((theta_2, t_2, size), (1, 0, size - 1)),
        ((theta_3, t_1, size), (1, 0, 0)),
        ((theta_3, t_0, size), (1, size - 1, 0)),
        ((theta_4, t_0, size), (2, 0, 0)),
        ((theta_4, t_1, size), (2, size - 1, 0)),
        ((theta_5, t_2, size), (2, 2 * size - 2, size - 1)),
        ((theta_5, t_3, size), (2, 0, size - 1)),
        ((theta_6, t_3, size), (3, 2 * size - 2, size - 1)),
        ((theta_6, t_2, size), (3, 0, size - 1)),
        ((theta_7, t_1, size), (3, 0, 0)),
        ((theta_7, t_0, size), (3, size - 1, 0)),
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
