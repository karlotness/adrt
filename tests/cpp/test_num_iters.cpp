/*
 * Copyright 2023 Karl Otness, Donsub Rim
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
#include "catch2/catch_amalgamated.hpp"
#include "adrt_cdefs_common.hpp"

using std::size_t;

TEST_CASE("num_iters handles zero", "[common][num_iters]") {
    const int result = adrt::num_iters(size_t{0});
    CHECK(result == 0);
}

TEST_CASE("num_iters handles small values", "[common][num_iters]") {
    size_t n = GENERATE(range(1, 17));
    const int expected = static_cast<int>(std::ceil(std::log2(static_cast<double>(n))));
    const int result = adrt::num_iters(n);
    CHECK(result == expected);
}

TEST_CASE("num_iters handles large value", "[common][num_iters]") {
    size_t n = (size_t{1} << (std::numeric_limits<size_t>::digits - 1)) + size_t{1};
    const int expected = std::numeric_limits<size_t>::digits;
    const int result = adrt::num_iters(n);
    CHECK(result == expected);
}

TEST_CASE("num_iters handles max value", "[common][num_iters]") {
    const size_t max_val = std::numeric_limits<size_t>::max();
    const int expected = std::numeric_limits<size_t>::digits;
    const int result = adrt::num_iters(max_val);
    CHECK(result == expected);
}
