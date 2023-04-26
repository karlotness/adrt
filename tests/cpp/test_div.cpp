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

#include <limits>
#include <cstddef>
#include <cmath>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("ceil_div gives correct result for small values", "[common][div][ceil_div]") {
    size_t val = GENERATE(range(0, 10));
    size_t d = GENERATE(1, range(3, 6));
    const size_t expected = static_cast<size_t>(std::ceil(static_cast<double>(val) / static_cast<double>(d)));
    CHECK(adrt::_common::ceil_div(val, d) == expected);
}

TEST_CASE("ceil_div2 gives correct result for small values", "[common][div][ceil_div2]") {
    size_t val = GENERATE(range(0, 10));
    const size_t expected = static_cast<size_t>(std::ceil(static_cast<double>(val) / 2.0));
    CHECK(adrt::_common::ceil_div2(val) == expected);
}

TEST_CASE("ceil_div2 gives correct result for max value",  "[common][div][ceil_div2]") {
    const size_t val = std::numeric_limits<size_t>::max();
    const size_t expected = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    CHECK(adrt::_common::ceil_div2(val) == expected);
}

TEST_CASE("floor_div gives correct result for small values", "[common][div][floor_div2]") {
    size_t val = GENERATE(range(0, 10));
    size_t d = GENERATE(1, range(3, 6));
    const size_t expected = static_cast<size_t>(std::floor(static_cast<double>(val) / static_cast<double>(d)));
    CHECK(adrt::_common::floor_div(val, d) == expected);
}

TEST_CASE("floor_div2 gives correct result for small values", "[common][div][floor_div2]") {
    size_t val = GENERATE(range(0, 10));
    const size_t expected = static_cast<size_t>(std::floor(static_cast<double>(val) / 2.0));
    CHECK(adrt::_common::floor_div2(val) == expected);
}

TEST_CASE("floor_div2 gives correct result for max value", "[common][div][floor_div2]") {
    const size_t val = std::numeric_limits<size_t>::max();
    const size_t expected = (size_t{1} << (std::numeric_limits<size_t>::digits - 1)) - size_t{1};
    CHECK(adrt::_common::floor_div2(val) == expected);
}
