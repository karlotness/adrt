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
#include <array>
#include <algorithm>
#include "catch2/catch_amalgamated.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("shape_product handles single elements without overflow", "[common][shape_product]") {
    size_t val = GENERATE(range(0, 5), std::numeric_limits<size_t>::max());

    SECTION("single-element array") {
        const std::array<size_t, 1> arr = {val};
        const auto result = adrt::_common::shape_product(arr);
        CHECK(result.value() == val);
    }

    SECTION("single scalar") {
        const auto result = adrt::_common::shape_product(&val, size_t{1});
        CHECK(result.value() == val);
    }
}

TEST_CASE("shape_product handles multi-element arrays without overflow", "[common][shape_product]") {
    std::array<size_t, 4> vals = {1, 2, 3, 4};
    REQUIRE(std::is_sorted(vals.cbegin(), vals.cend()));
    do {
        const auto result = adrt::_common::shape_product(vals);
        CHECK(result.value() == size_t{24});
    } while(std::next_permutation(vals.begin(), vals.end()));
}

TEST_CASE("shape_product handles arrays with max product", "[common][shape_product]") {
    const size_t max_val = std::numeric_limits<size_t>::max();
    std::array<size_t, 4> vals = {1, 1, 3, max_val / size_t{3}};
    REQUIRE(std::is_sorted(vals.cbegin(), vals.cend()));
    if(max_val % size_t{3} == size_t{0}) {
        do {
            const auto result = adrt::_common::shape_product(vals);
            CHECK(result.value() == max_val);
        } while(std::next_permutation(vals.begin(), vals.end()));
    }
    else {
        WARN("Unusual bit-ness for platform, test skipped");
    }
}

TEST_CASE("shape_product handles empty arrays", "[common][shape_product]") {
    SECTION("non-null pointer") {
        const size_t val = 0;
        const auto result = adrt::_common::shape_product(&val, size_t{0});
        CHECK_FALSE(result.has_value());
    }

    SECTION("null pointer") {
        const auto result = adrt::_common::shape_product(nullptr, size_t{0});
        CHECK_FALSE(result.has_value());
    }

    SECTION("zero-size array") {
        const std::array<size_t, 0> arr;
        const auto result = adrt::_common::shape_product(arr);
        CHECK_FALSE(result.has_value());
    }
}

TEST_CASE("shape_product handles arrays with overflow", "[common][shape_product]") {
    const size_t size_t_half = size_t{1} << (std::numeric_limits<size_t>::digits / 2);
    std::array<size_t, 4> vals = {1, 1, size_t_half, size_t_half << (std::numeric_limits<size_t>::digits % 2)};
    REQUIRE(std::is_sorted(vals.cbegin(), vals.cend()));
    do {
        const auto result = adrt::_common::shape_product(vals);
        CHECK_FALSE(result.has_value());
    } while(std::next_permutation(vals.begin(), vals.end()));
}

TEST_CASE("shape_product is commutative with overflow and zero", "[common][shape_product]") {
    const size_t size_t_half = size_t{1} << (std::numeric_limits<size_t>::digits / 2);
    std::array<size_t, 4> vals = {0, 1, size_t_half, size_t_half << (std::numeric_limits<size_t>::digits % 2)};
    REQUIRE(std::is_sorted(vals.cbegin(), vals.cend()));
    do {
        const auto result = adrt::_common::shape_product(vals);
        CHECK(result.value() == size_t{0});
    } while(std::next_permutation(vals.begin(), vals.end()));
}
