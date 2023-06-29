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
#include <array>
#include <limits>
#include "catch2/catch_amalgamated.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

namespace adrt_test {
    namespace {
        template<size_t N>
        std::array<unsigned int, N> array_sequence() {
            constexpr unsigned int val_increment = 0xab00;
            static_assert(N < std::numeric_limits<unsigned int>::max(), "Max value for array too large");
            static_assert(val_increment < std::numeric_limits<unsigned int>::max() - N, "Increment for tests too large");
            std::array<unsigned int, N> arr;
            for(unsigned int i = 0; i < N; ++i) {
                arr.at(i) = val_increment + i;
            }
            return arr;
        }
    }
}

TEST_CASE("array_access handles 1D shape", "[common][array_access]") {
    const std::array<size_t, 1> shape = {8};
    const auto arr = adrt_test::array_sequence<8>();
    for(size_t i = 0; i < std::get<0>(shape); ++i) {
        CHECK(adrt::_common::array_access(arr.data(), shape, i) == arr.at(i));
    }
}

TEST_CASE("array_access handles 2D shape", "[common][array_access]") {
    const std::array<size_t, 2> shape = {3, 4};
    const auto arr = adrt_test::array_sequence<12>();
    for(size_t i = 0; i < std::get<0>(shape); ++i) {
        for(size_t j = 0; j < std::get<1>(shape); ++j) {
            CHECK(adrt::_common::array_access(arr.data(), shape, i, j) == arr.at(i * std::get<1>(shape) + j));
        }
    }
}

TEST_CASE("array_access handles 3D shape", "[common][array_access]") {
    const std::array<size_t, 3> shape = {2, 3, 4};
    const auto arr = adrt_test::array_sequence<24>();
    for(size_t i = 0; i < std::get<0>(shape); ++i) {
        for(size_t j = 0; j < std::get<1>(shape); ++j) {
            for(size_t k = 0; k < std::get<2>(shape); ++k) {
                CHECK(adrt::_common::array_access(arr.data(), shape, i, j, k) == arr.at(i * std::get<1>(shape) * std::get<2>(shape) + j * std::get<2>(shape) + k));
            }
        }
    }
}

TEST_CASE("array_access assigns 3D shape", "[common][array_access]") {
    const unsigned int max_val = std::numeric_limits<unsigned int>::max();
    const std::array<size_t, 3> shape = {2, 3, 4};
    auto arr = adrt_test::array_sequence<24>();
    REQUIRE(std::get<21>(arr) != max_val);
    adrt::_common::array_access(arr.data(), shape, size_t{1}, size_t{2}, size_t{1}) = max_val;
    CHECK(std::get<21>(arr) == max_val);
}
