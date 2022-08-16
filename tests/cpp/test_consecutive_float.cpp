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

#include <cmath>
#include "catch2/catch.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("max consecutive size_t is exactly represented in float", "[common][const][float][float_consecutive_size_t]") {
    const size_t max_float_size_t = adrt::_const::largest_consecutive_float_size_t<float>();
    const float float_val = max_float_size_t;
    CHECK(std::trunc(float_val) == float_val);
    CHECK(max_float_size_t == static_cast<size_t>(float_val));
}

TEST_CASE("max consecutive size_t is exactly represented in double", "[common][const][float][float_consecutive_size_t]") {
    const size_t max_double_size_t = adrt::_const::largest_consecutive_float_size_t<double>();
    const double double_val = max_double_size_t;
    CHECK(std::trunc(double_val) == double_val);
    CHECK(max_double_size_t == static_cast<size_t>(double_val));
}

TEST_CASE("max consecutive size_t float is consecutive below", "[common][const][float][float_consecutive_size_t]") {
    const float float_val = adrt::_const::largest_consecutive_float_size_t<float>();
    const float float_val_below = float_val - 1.0f;
    CHECK(std::trunc(float_val_below) == float_val_below);
    CHECK(float_val - float_val_below == 1.0f);
}

TEST_CASE("max consecutive size_t double is consecutive below", "[common][const][float][float_consecutive_size_t]") {
    const double double_val = adrt::_const::largest_consecutive_float_size_t<double>();
    const double double_val_below = double_val - 1.0;
    CHECK(std::trunc(double_val_below) == double_val_below);
    CHECK(double_val - double_val_below == 1.0);
}

TEST_CASE("max consecutive size_t float is not consecutive above", "[common][const][float][float_consecutive_size_t]") {
    const float float_val = adrt::_const::largest_consecutive_float_size_t<float>();
    const float float_val_above = float_val + 1.0f;
    CHECK(float_val == float_val_above);
}

TEST_CASE("max consecutive size_t double is not consecutive above", "[common][const][float][float_consecutive_size_t]") {
    const double double_val = adrt::_const::largest_consecutive_float_size_t<double>();
    const double double_val_above = double_val + 1.0;
    CHECK(double_val == double_val_above);
}
