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

#include <cstddef>
#include <cmath>
#include <tuple>
#include <limits>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;
using float_test_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_CASE("max consecutive size_t is exactly represented in floating point", "[common][const][float][float_consecutive_size_t]", float_test_types) {
    const size_t max_float_size_t = adrt::_const::largest_consecutive_float_size_t<TestType>;
    const TestType float_val = max_float_size_t;
    REQUIRE(std::isfinite(float_val));
    CHECK(std::trunc(float_val) == float_val);
    CHECK(max_float_size_t == static_cast<size_t>(float_val));
}

TEMPLATE_LIST_TEST_CASE("max consecutive size_t floating point is consecutive below", "[common][const][float][float_consecutive_size_t]", float_test_types) {
    const TestType float_val = adrt::_const::largest_consecutive_float_size_t<TestType>;
    const TestType float_val_below = float_val - static_cast<TestType>(1);
    CHECK(std::trunc(float_val_below) == float_val_below);
    CHECK(float_val - float_val_below == static_cast<TestType>(1));
}

TEMPLATE_LIST_TEST_CASE("max consecutive size_t is likely at end of consecutive range", "[common][const][float][float_consecutive_size_t]", float_test_types) {
    const TestType float_val = adrt::_const::largest_consecutive_float_size_t<TestType>;
    const TestType float_val_above = float_val + static_cast<TestType>(1);
    if(std::numeric_limits<TestType>::digits < std::numeric_limits<size_t>::digits) {
        // Normal float type, non-consecutive above
        CHECK(float_val == float_val_above);
    }
    else {
        // Very big float type (size_t max fits into consecutive range)
        CHECK(float_val == static_cast<TestType>(std::numeric_limits<size_t>::max()));
        CHECK(std::trunc(float_val_above) == float_val_above);
        CHECK(float_val_above - float_val == static_cast<TestType>(1));
    }
}
