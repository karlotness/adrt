/*
 * Copyright Karl Otness, Donsub Rim
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cmath>
#include <tuple>
#include <limits>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using float_test_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_CASE("lerp with opposite signs outputs positive value at zero", "[common][float][lerp][lerp_opposite_signs]", float_test_types) {
    const TestType val = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(val, -val, static_cast<TestType>(0));
    CHECK(lerp == std::abs(val));
}

TEMPLATE_LIST_TEST_CASE("lerp with opposite signs outputs negative value at one", "[common][float][lerp][lerp_opposite_signs]", float_test_types) {
    const TestType val = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(val, -val, static_cast<TestType>(1));
    CHECK(lerp == -std::abs(val));
}

TEMPLATE_LIST_TEST_CASE("lerp with opposite signs outputs zero at half", "[common][float][lerp][lerp_opposite_signs]", float_test_types) {
    const TestType val = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(val, -val, static_cast<TestType>(0.5L));
    CHECK(lerp == static_cast<TestType>(0));
}

TEMPLATE_LIST_TEST_CASE("lerp with opposite signs outputs positive half value at a quarter", "[common][float][lerp][lerp_opposite_signs]", float_test_types) {
    const TestType val = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(val, -val, static_cast<TestType>(0.25L));
    CHECK(lerp == static_cast<TestType>(1));
}

TEMPLATE_LIST_TEST_CASE("lerp with opposite signs outputs negative half value at three quarters", "[common][float][lerp][lerp_opposite_signs]", float_test_types) {
    const TestType val = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(val, -val, static_cast<TestType>(0.75L));
    CHECK(lerp == -static_cast<TestType>(1));
}

TEMPLATE_LIST_TEST_CASE("lerp with same signs outputs left value at zero", "[common][float][lerp][lerp_same_signs]", float_test_types) {
    const TestType left = static_cast<TestType>(1);
    const TestType right = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(left, right, static_cast<TestType>(0));
    CHECK(lerp == left);
}

TEMPLATE_LIST_TEST_CASE("lerp with same signs outputs right value at one", "[common][float][lerp][lerp_same_signs]", float_test_types) {
    const TestType left = static_cast<TestType>(1);
    const TestType right = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(left, right, static_cast<TestType>(1));
    CHECK(lerp == right);
}

TEMPLATE_LIST_TEST_CASE("lerp with same signs outputs mean at half", "[common][float][lerp][lerp_same_signs]", float_test_types) {
    const TestType left = static_cast<TestType>(1);
    const TestType right = static_cast<TestType>(2);
    const TestType lerp = adrt::_common::lerp(left, right, static_cast<TestType>(0.5L));
    CHECK(lerp == static_cast<TestType>(1.5L));
}

TEMPLATE_LIST_TEST_CASE("lerp propagates NaN", "[common][float][lerp][lerp_nan]", float_test_types) {
    static_assert(std::numeric_limits<TestType>::has_quiet_NaN, "Test requires valid quiet NaN value");
    const TestType nan = std::numeric_limits<TestType>::quiet_NaN();
    TestType t_val = static_cast<TestType>(GENERATE(0.0L, 0.5L, 1.0L));
    CHECK(std::isnan(adrt::_common::lerp(nan, nan, t_val)));
    CHECK(std::isnan(adrt::_common::lerp(nan, static_cast<TestType>(0.5L), t_val)));
    CHECK(std::isnan(adrt::_common::lerp(static_cast<TestType>(0.5L), nan, t_val)));
}

TEMPLATE_LIST_TEST_CASE("lerp propagates NaN with same signs", "[common][float][lerp][lerp_same_signs]", float_test_types) {
    static_assert(std::numeric_limits<TestType>::has_quiet_NaN, "Test requires valid quiet NaN value");
    const TestType nan = std::numeric_limits<TestType>::quiet_NaN();
    const TestType left = static_cast<TestType>(1);
    const TestType right = static_cast<TestType>(2);
    CHECK(std::isnan(adrt::_common::lerp(left, right, nan)));
}

TEMPLATE_LIST_TEST_CASE("lerp propagates NaN with opposite signs", "[common][float][lerp][lerp_same_signs]", float_test_types) {
    static_assert(std::numeric_limits<TestType>::has_quiet_NaN, "Test requires valid quiet NaN value");
    const TestType nan = std::numeric_limits<TestType>::quiet_NaN();
    const TestType val = static_cast<TestType>(2);
    CHECK(std::isnan(adrt::_common::lerp(-val, val, nan)));
}
