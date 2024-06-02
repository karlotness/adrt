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

#include <span>
#include <array>
#include "catch2/catch_amalgamated.hpp"
#include "adrt_cdefs_common.hpp"

TEST_CASE("get with const span", "[common][span][get]") {
    std::array<size_t, 3> arr = {1, 2, 3};
    std::span<const size_t, size_t{3}> tmpspan{arr};
    STATIC_CHECK(std::is_same_v<decltype(adrt::_common::get<0>(tmpspan)), decltype(tmpspan[0])>);
    CHECK(adrt::_common::get<0>(tmpspan) == size_t{1});
    CHECK(adrt::_common::get<2>(tmpspan) == size_t{3});
}

TEST_CASE("get with span", "[common][span][get]") {
    std::array<int, 3> arr = {1, 2, 3};
    std::span<int, size_t{3}> tmpspan{arr};
    STATIC_CHECK(std::is_same_v<decltype(adrt::_common::get<0>(tmpspan)), decltype(tmpspan[0])>);
    CHECK(adrt::_common::get<0>(tmpspan) == 1);
    CHECK(adrt::_common::get<2>(tmpspan) == 3);
}

TEST_CASE("get with const array", "[common][span][get]") {
    const std::array<size_t, 3> arr = {1, 2, 3};
    STATIC_CHECK(std::is_same_v<decltype(adrt::_common::get<0>(arr)), decltype(std::get<0>(arr))>);
    CHECK(adrt::_common::get<0>(arr) == size_t{1});
    CHECK(adrt::_common::get<2>(arr) == size_t{3});
}

TEST_CASE("get with array", "[common][span][get]") {
    std::array<int, 3> arr = {1, 2, 3};
    STATIC_CHECK(std::is_same_v<decltype(adrt::_common::get<0>(arr)), decltype(std::get<0>(arr))>);
    CHECK(adrt::_common::get<0>(arr) == 1);
    CHECK(adrt::_common::get<2>(arr) == 3);
}
