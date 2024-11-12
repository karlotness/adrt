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

#include <cstddef>
#include <limits>
#include <array>
#include <span>
#include <cassert>
#include <optional>
#include <algorithm>
#include <bit>

#ifdef _MSC_VER
// MSVC intrinsics
#include <intrin.h>
#endif

#include "adrt_cdefs_common.hpp"
#include "adrt_cdefs_adrt.hpp"
#include "adrt_cdefs_iadrt.hpp"
#include "adrt_cdefs_bdrt.hpp"
#include "adrt_cdefs_interp_adrtcart.hpp"
#include "adrt_cdefs_fmg.hpp"

using namespace adrt::_literals;
using std::size_t;

namespace adrt { namespace _impl { namespace {

// This imposes a limit on max_iters(n) <= (digits-1) and ensures that in our
// single-step functions the shifts of 1<<(iter+1) never go out of range
const size_t max_size = 1_uz << (std::numeric_limits<size_t>::digits - 1);

bool is_pow2(size_t val) {
    if(val == 0u) {
        return false;
    }
    return (val & (val - 1_uz)) == 0u;
}


[[maybe_unused, nodiscard]] bool mul_check_fallback(size_t a, size_t b, size_t &prod) {
    prod = a * b;
    const bool overflow = (b != 0u) && (a > std::numeric_limits<size_t>::max() / b);
    return !overflow;
}

template <size_t N>
std::array<size_t, N> span_to_array(std::span<const size_t, N> s) {
    static_assert(N != std::dynamic_extent, "Span must have statically-known extent");
    return [&s]<size_t... Ints>(std::index_sequence<Ints...>) -> std::array<size_t, N> {
        return {adrt::_common::get<Ints>(s)...};
    }(std::make_index_sequence<N>{});
}

#if defined(__GNUC__) || defined(__clang__) // GCC intrinsics

// Compatibility with old GCC
#ifndef __has_builtin
#define __has_builtin(feat) 0
#endif

[[nodiscard]] bool mul_check(size_t a, size_t b, size_t &prod) {

    #if __has_builtin(__builtin_mul_overflow) || (defined(__GNUC__) && (__GNUC__ >= 5))
    {
        const bool overflow = __builtin_mul_overflow(a, b, &prod);
        return !overflow;
    }
    #endif

    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#elif defined(_MSC_VER) // MSVC intrinsics

[[nodiscard]] bool mul_check(size_t a, size_t b, size_t &prod) {

    #if (_MSC_VER >= 1937) && (defined(_M_IX86) || defined(_M_X64))
    if constexpr(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned int>::max()) {
        const unsigned int ua = static_cast<unsigned int>(a);
        const unsigned int ub = static_cast<unsigned int>(b);
        unsigned int res_low, res_high;
        const bool overflow = static_cast<bool>(_mul_full_overflow_u32(ua, ub, &res_low, &res_high));
        assert(overflow == (res_high != 0u));
        prod = static_cast<size_t>(res_low);
        return !overflow && (res_low <= std::numeric_limits<size_t>::max());
    }
    #endif

    #if (_MSC_VER >= 1937) && defined(_M_X64)
    if constexpr(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned __int64>::max()) {
        const unsigned __int64 ua = static_cast<unsigned __int64>(a);
        const unsigned __int64 ub = static_cast<unsigned __int64>(b);
        unsigned __int64 res_low, res_high;
        const bool overflow = static_cast<bool>(_mul_full_overflow_u64(ua, ub, &res_low, &res_high));
        assert(overflow == (res_high != 0u));
        prod = static_cast<size_t>(res_low);
        return !overflow && (res_low <= std::numeric_limits<size_t>::max());
    }
    #endif

    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#else // Fallback only

[[nodiscard]] bool mul_check(size_t a, size_t b, size_t &prod) {
    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#endif // End platform cases

}}} // End namespace adrt::_impl

namespace adrt {

    int num_iters(size_t shape) {
        return static_cast<int>(std::bit_width(shape)) - (adrt::_impl::is_pow2(shape) ? 1 : 0);
    }

    namespace _common {
        std::optional<size_t> mul_check(size_t a, size_t b) {
            size_t prod;
            const bool ok = adrt::_impl::mul_check(a, b, prod);
            if(!ok) {
                return {};
            }
            return {prod};
        }

        std::optional<size_t> shape_product(std::span<const size_t> shape) {
            assert(shape.empty() || (shape.data() != nullptr));
            if(shape.empty()) {
                return {};
            }
            std::optional<size_t> prod = shape[0];
            for(size_t i = 1; i < shape.size(); ++i) {
                if(prod) {
                    prod = adrt::_common::mul_check(*prod, shape[i]);
                }
                else if(shape[i] == 0u) {
                    // We don't anticipate zero shapes, but this makes
                    // shape_product commutative.
                    prod = 0_uz;
                }
            }
            return prod;
        }

    } // End adrt::_common

    // Implementation for adrt
    bool adrt_is_valid_shape(std::span<const size_t, 3> shape) {
        // Make sure array is square
        return (std::ranges::all_of(shape, [](size_t v){return v > 0u;}) && // All entries must be nonzero
                (adrt::_common::get<1>(shape) == adrt::_common::get<2>(shape)) && // Must be square
                (adrt::_common::get<2>(shape) <= adrt::_impl::max_size) &&
                (adrt::_impl::is_pow2(adrt::_common::get<2>(shape)))); // Must have power of two shape
    }

    // Implementation for adrt
    bool adrt_step_is_valid_shape(std::span<const size_t, 4> shape) {
        // Check if the rows & cols are shaped like an ADRT output
        return (std::ranges::all_of(shape, [](size_t v){return v > 0u;}) &&
                (adrt::_common::get<1>(shape) == 4u) &&
                (adrt::_common::get<3>(shape) <= adrt::_impl::max_size) &&
                (adrt::_common::get<2>(shape) == (adrt::_common::get<3>(shape) * 2_uz - 1_uz)) &&
                (adrt::_impl::is_pow2(adrt::_common::get<3>(shape))));
    }

    // Implementation for adrt
    bool adrt_step_is_valid_iter(std::span<const size_t, 4> shape, int iter) {
        return iter >= 0 && iter < adrt::num_iters(adrt::_common::get<3>(shape));
    }

    // Implementation for adrt
    std::array<size_t, 5> adrt_buffer_shape(std::span<const size_t, 3> shape) {
        return {
            adrt::_common::get<0>(shape),
            4,
            adrt::_common::get<1>(shape),
            1,
            2_uz * adrt::_common::get<2>(shape) - 1_uz, // No overflow because n^2 fits in size_t, so must 2*n
        };
    }

    // Implementation for adrt
    std::array<size_t, 4> adrt_result_shape(std::span<const size_t, 3> shape) {
        return {
            adrt::_common::get<0>(shape),
            4,
            2_uz * adrt::_common::get<2>(shape) - 1_uz, // No overflow because n^2 fits in size_t, so must 2*n
            adrt::_common::get<1>(shape)
        };
    }

    std::array<size_t, 4> adrt_step_result_shape(std::span<const size_t, 4> shape) {
        return adrt::_impl::span_to_array(shape);
    }

    // Implementation for bdrt
    bool bdrt_is_valid_shape(std::span<const size_t, 4> shape) {
        return adrt::adrt_step_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_shape(std::span<const size_t, 4> shape) {
        return adrt::bdrt_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_iter(std::span<const size_t, 4> shape, int iter) {
        return adrt::adrt_step_is_valid_iter(shape, iter);
    }

    // Implementation for bdrt
    std::array<size_t, 5> bdrt_buffer_shape(std::span<const size_t, 4> shape) {
        return {
            adrt::_common::get<0>(shape), // batch
            4,  // quadrant
            adrt::_common::get<3>(shape), // col
            1, // sections
            adrt::_common::get<2>(shape), // row
        };
    }

    // Implementation for bdrt
    std::array<size_t, 4> bdrt_result_shape(std::span<const size_t, 4> shape) {
        return adrt::_impl::span_to_array(shape);
    }

    std::array<size_t, 4> bdrt_step_result_shape(std::span<const size_t, 4> shape) {
        return adrt::_impl::span_to_array(shape);
    }

    bool iadrt_is_valid_shape(std::span<const size_t, 4> shape) {
        // bdrt also requires its input to have the shape of an adrt result, reuse
        return adrt::bdrt_is_valid_shape(shape);
    }

    std::array<size_t, 5> iadrt_buffer_shape(std::span<const size_t, 4> shape) {
        return {
            adrt::_common::get<0>(shape), // planes
            4, // Quadrants (shape[1])
            1,
            adrt::_common::get<3>(shape), // N
            adrt::_common::get<2>(shape), // 2 * N - 1
        };
    }

    std::array<size_t, 4> iadrt_result_shape(std::span<const size_t, 4> shape) {
        return adrt::_impl::span_to_array(shape);
    }

    bool interp_adrtcart_is_valid_shape(std::span<const size_t, 4> shape) {
        // bdrt also requires its input to have the shape of an adrt result, reuse
        return adrt::bdrt_is_valid_shape(shape) && adrt::_common::get<3>(shape) > 1u;
    }

    std::array<size_t, 3> interp_adrtcart_result_shape(std::span<const size_t, 4> shape) {
        return {
            adrt::_common::get<0>(shape), // batch
            adrt::_common::get<3>(shape), // rows
            4_uz * adrt::_common::get<3>(shape), // cols. No overflow, merges quadrant and column dimensions
        };
    }

    bool fmg_restriction_is_valid_shape(std::span<const size_t, 4> shape) {
        return (std::ranges::all_of(shape, [](size_t v){return v > 0u;}) &&
                adrt::_common::get<1>(shape) == 4u &&
                adrt::_common::get<3>(shape) <= adrt::_impl::max_size &&
                adrt::_common::get<2>(shape) == (adrt::_common::get<3>(shape) * 2_uz - 1_uz) &&
                adrt::_common::get<3>(shape) >= 2u &&
                adrt::_common::get<3>(shape) % 2_uz == 0u);
    }

    std::array<size_t, 4> fmg_restriction_result_shape(std::span<const size_t, 4> shape) {
        return {
            adrt::_common::get<0>(shape), // batches
            4, // quadrants
            adrt::_common::get<3>(shape) - 1_uz, // rows
            adrt::_common::get<3>(shape) / 2_uz, // cols (halved)
        };
    }

    bool fmg_prolongation_is_valid_shape(std::span<const size_t, 3> shape) {
        const size_t prl_max_size = adrt::_common::floor_div2(std::numeric_limits<size_t>::max());
        return (std::ranges::all_of(shape, [](size_t v){return v > 0u;}) &&
                adrt::_common::get<1>(shape) <= prl_max_size &&
                adrt::_common::get<2>(shape) <= prl_max_size);
    }

    std::array<size_t, 3> fmg_prolongation_result_shape(std::span<const size_t, 3> shape) {
        return {
            adrt::_common::get<0>(shape), // batch
            2_uz * adrt::_common::get<1>(shape), // rows. No overflow, checked in fmg_prolongation_is_valid_shape
            2_uz * adrt::_common::get<2>(shape), // cols
        };
    }

    bool fmg_highpass_is_valid_shape(std::span<const size_t, 3> shape) {
        return (std::ranges::all_of(shape, [](size_t v){return v > 0u;}) &&
                adrt::_common::get<1>(shape) >= 2u &&
                adrt::_common::get<2>(shape) >= 2u);
    }

    std::array<size_t, 3> fmg_highpass_result_shape(std::span<const size_t, 3> shape) {
        return adrt::_impl::span_to_array(shape);
    }

} // End namespace adrt
