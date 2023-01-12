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

#include <type_traits>
#include <array>
#include <cstddef>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("Optional is trivially-destructible for trivial types", "[common][optional]") {
    STATIC_REQUIRE(std::is_trivially_destructible<adrt::_common::Optional<int>>::value);
    STATIC_REQUIRE(std::is_trivially_destructible<adrt::_common::Optional<size_t>>::value);
    STATIC_REQUIRE(std::is_trivially_destructible<adrt::_common::Optional<std::array<size_t, 5>>>::value);
}

TEST_CASE("Optional default constructor is empty", "[common][optional]") {
    const adrt::_common::Optional<int> opt;
    REQUIRE_FALSE(opt.has_value());
}

TEST_CASE("Optional assignment constructor stores value", "[common][optional]") {
    const adrt::_common::Optional<int> opt = 5;
    REQUIRE(opt.has_value());
    CHECK(*opt == 5);
}

TEST_CASE("Optional construction stores value", "[common][optional]") {
    const adrt::_common::Optional<int> opt(5);
    REQUIRE(opt.has_value());
    CHECK(*opt == 5);
}
