/*
 * Copyright (c) 2023 Karl Otness, Donsub Rim
 * All rights reserved
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

TEMPLATE_LIST_TEST_CASE("clamp limits floats below range", "[common][float][clamp]", float_test_types) {
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    const TestType val = std::nextafter(lower, -static_cast<TestType>(2));
    REQUIRE(val < lower);
    const TestType clamped = adrt::_common::clamp(val, lower, higher);
    CHECK(clamped == lower);
}

TEMPLATE_LIST_TEST_CASE("clamp limits floats above range", "[common][float][clamp]", float_test_types) {
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    const TestType val = std::nextafter(higher, static_cast<TestType>(2));
    REQUIRE(val > higher);
    const TestType clamped = adrt::_common::clamp(val, lower, higher);
    CHECK(clamped == higher);
}

TEMPLATE_LIST_TEST_CASE("clamp leaves lower limit alone", "[common][float][clamp]", float_test_types) {
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    const TestType clamped = adrt::_common::clamp(lower, lower, higher);
    CHECK(clamped == lower);
}

TEMPLATE_LIST_TEST_CASE("clamp leaves upper limit alone", "[common][float][clamp]", float_test_types) {
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    const TestType clamped = adrt::_common::clamp(higher, lower, higher);
    CHECK(clamped == higher);
}

TEMPLATE_LIST_TEST_CASE("clamp leaves floats within range alone", "[common][float][clamp]", float_test_types) {
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    int numerator = GENERATE(-1, 0, 1);
    const TestType val = static_cast<TestType>(numerator) / static_cast<TestType>(2);
    REQUIRE(val > lower);
    REQUIRE(val < higher);
    const TestType clamped = adrt::_common::clamp(val, lower, higher);
    CHECK(clamped == val);
}

TEMPLATE_LIST_TEST_CASE("clamp propagates NaN values", "[common][float][clamp]", float_test_types) {
    static_assert(std::numeric_limits<TestType>::has_quiet_NaN, "Test requires valid quiet NaN value");
    const TestType nan = std::numeric_limits<TestType>::quiet_NaN();
    const TestType lower = -static_cast<TestType>(1);
    const TestType higher = static_cast<TestType>(1);
    CHECK(std::isnan(adrt::_common::clamp(nan, lower, higher)));
}
