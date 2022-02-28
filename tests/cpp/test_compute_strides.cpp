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

#include <cstddef>
#include <array>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("compute_strides handles 1D shape", "[common][compute_strides]") {
    const std::array<size_t, 1> shape = {5};
    const std::array<size_t, 1> expected_strides = {1};
    const auto strides = adrt::_common::compute_strides(shape);
    REQUIRE(strides == expected_strides);
}

TEST_CASE("compute_strides handles 2D shape", "[common][compute_strides]") {
    const std::array<size_t, 2> shape = {5, 4};
    const std::array<size_t, 2> expected_strides = {4, 1};
    const auto strides = adrt::_common::compute_strides(shape);
    REQUIRE(strides == expected_strides);
}

TEST_CASE("compute_strides handles 3D shape", "[common][compute_strides]") {
    const std::array<size_t, 3> shape = {5, 4, 3};
    const std::array<size_t, 3> expected_strides = {12, 3, 1};
    const auto strides = adrt::_common::compute_strides(shape);
    REQUIRE(strides == expected_strides);
}

TEST_CASE("compute_strides handles 4D shape", "[common][compute_strides]") {
    const std::array<size_t, 4> shape = {5, 4, 3, 2};
    const std::array<size_t, 4> expected_strides = {24, 6, 2, 1};
    const auto strides = adrt::_common::compute_strides(shape);
    REQUIRE(strides == expected_strides);
}
