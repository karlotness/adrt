/*
 * Copyright (c) 2022 Karl Otness, Donsub Rim
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

#include <limits>
#include <cstddef>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("mul_check computes small products without overflow", "[common][mul_check]") {
    size_t a = GENERATE(range(0, 4));
    size_t b = GENERATE(range(0, 4));
    auto result = adrt::_common::mul_check(a, b);
    REQUIRE(result.has_value());
    CHECK(*result == (a * b));
}

TEST_CASE("mul_check computes products with size_t_max and 1", "[common][mul_check]") {
    size_t max_val = std::numeric_limits<size_t>::max();

    SECTION("order a") {
        auto result = adrt::_common::mul_check(max_val, 1);
        REQUIRE(result.has_value());
        CHECK(*result == max_val);
    }

    SECTION("order b") {
        auto result = adrt::_common::mul_check(1, max_val);
        REQUIRE(result.has_value());
        CHECK(*result == max_val);
    }
}

TEST_CASE("mul_check computes products with size_t_max and 0", "[common][mul_check]") {
    size_t max_val = std::numeric_limits<size_t>::max();

    SECTION("order a") {
        auto result = adrt::_common::mul_check(max_val, 0);
        REQUIRE(result.has_value());
        CHECK(*result == size_t{0});
    }

    SECTION("order b") {
        auto result = adrt::_common::mul_check(0, max_val);
        REQUIRE(result.has_value());
        CHECK(*result == size_t{0});
    }
}

TEST_CASE("mul_check correctly produces size_t_max", "[common][mul_check]") {
    const size_t max_val = std::numeric_limits<size_t>::max();
    if(max_val % size_t{3} == size_t{0}) {
        // Test can proceed (should apply to 32-/64-bit platforms)
        size_t max_3 = max_val / size_t{3};

        SECTION("order a") {
            auto result = adrt::_common::mul_check(max_3, size_t{3});
            REQUIRE(result.has_value());
            CHECK(*result == max_val);
        }

        SECTION("order b") {
            auto result = adrt::_common::mul_check(size_t{3}, max_3);
            REQUIRE(result.has_value());
            CHECK(*result == max_val);
        }
    }
    else {
        WARN("Unusual bit-ness for platform, test skipped");
    }
}

TEST_CASE("mul_check detects overflow with sqrt(size_t_max)", "[common][mul_check]") {
    size_t size_t_half = size_t{1} << (std::numeric_limits<size_t>::digits / 2);
    auto result = adrt::_common::mul_check(size_t_half, size_t_half);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("mul_check detects overflow with size_t_max", "[common][mul_check]") {
    size_t max_val = std::numeric_limits<size_t>::max();
    auto result = adrt::_common::mul_check(max_val, max_val);
    REQUIRE_FALSE(result.has_value());
}
