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
#include <array>
#include <cassert>

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

int num_iters_fallback(size_t shape) {
    // Relies on earlier check that shape != 0
    assert(shape != 0u);
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    int r = 0;
    while(shape != 0) {
        ++r;
        shape >>= 1;
    }
    return r + (is_power_of_two ? 0 : 1) - 1;
}

bool mul_check_fallback(size_t a, size_t b, size_t &prod) {
    prod = a * b;
    const bool overflow = (b != 0u) && (a > std::numeric_limits<size_t>::max() / b);
    return !overflow;
}

template<size_t N>
bool all_positive(const std::array<size_t, N> &shape) {
    // Make sure all shapes are nonzero
    for(size_t i = 0; i < shape.size(); ++i) {
        if(shape[i] <= 0u) {
            return false;
        }
    }
    return true;
}

// Implementation of adrt_num_iters

#if defined(__GNUC__) || defined(__clang__) // GCC intrinsics

// Compatibility with old GCC
#ifndef __has_builtin
#define __has_builtin(feat) 0
#endif

int num_iters(size_t shape) {
    // Relies on earlier check that shape != 0
    assert(shape != 0u);
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned int>::max()) {
        const unsigned int ushape = static_cast<unsigned int>(shape);
        const int lead_zero = __builtin_clz(ushape);
        return (std::numeric_limits<unsigned int>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long>::max()) {
        const unsigned long ushape = static_cast<unsigned long>(shape);
        const int lead_zero = __builtin_clzl(ushape);
        return (std::numeric_limits<unsigned long>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long long>::max()) {
        const unsigned long long ushape = static_cast<unsigned long long>(shape);
        const int lead_zero = __builtin_clzll(ushape);
        return (std::numeric_limits<unsigned long long>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    return adrt::_impl::num_iters_fallback(shape);
}

bool mul_check(size_t a, size_t b, size_t &prod) {

    #if __has_builtin(__builtin_mul_overflow) || (defined(__GNUC__) && (__GNUC__ >= 5))
    {
        const bool overflow = __builtin_mul_overflow(a, b, &prod);
        return !overflow;
    }
    #endif

    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#elif defined(_MSC_VER) // MSVC intrinsics

int num_iters(size_t shape) {
    // Relies on earlier check that shape != 0
    assert(shape != 0u);
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long>::max()) {
        unsigned long index;
        const unsigned long ushape = static_cast<unsigned long>(shape);
        _BitScanReverse(&index, ushape);
        return static_cast<int>(index) + (is_power_of_two ? 0 : 1);
    }

    #if defined(_M_X64) || defined(_M_ARM64)
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned __int64>::max()) {
        unsigned long index;
        const unsigned __int64 ushape = static_cast<unsigned __int64>(shape);
        _BitScanReverse64(&index, ushape);
        return static_cast<int>(index) + (is_power_of_two ? 0 : 1);
    }
    #endif // End: 64bit arch

    return adrt::_impl::num_iters_fallback(shape);
}

bool mul_check(size_t a, size_t b, size_t &prod) {
    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#else // Fallback only

int num_iters(size_t shape) {
    return adrt::_impl::num_iters_fallback(shape);
}

bool mul_check(size_t a, size_t b, size_t &prod) {
    return adrt::_impl::mul_check_fallback(a, b, prod);
}

#endif // End platform cases

}}} // End namespace adrt::_impl

namespace adrt {

    int num_iters(size_t shape) {
        if(shape == 0u) {
            return 0;
        }
        return adrt::_impl::num_iters(shape);
    }

    namespace _common {
        adrt::_common::Optional<size_t> mul_check(size_t a, size_t b) {
            size_t prod;
            const bool ok = adrt::_impl::mul_check(a, b, prod);
            if(!ok) {
                return {};
            }
            return {prod};
        }

        adrt::_common::Optional<size_t> shape_product(const size_t *shape, size_t n) {
            assert((n == 0u) || shape);
            if(n == 0u) {
                return {};
            }
            adrt::_common::Optional<size_t> prod = shape[0];
            for(size_t i = 1; i < n; ++i) {
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
    bool adrt_is_valid_shape(const std::array<size_t, 3> &shape) {
        // Make sure array is square
        return (adrt::_impl::all_positive(shape) && // All entries must be nonzero
                (std::get<1>(shape) == std::get<2>(shape)) && // Must be square
                (std::get<2>(shape) <= adrt::_impl::max_size) &&
                (adrt::_impl::is_pow2(std::get<2>(shape)))); // Must have power of two shape
    }

    // Implementation for adrt
    bool adrt_step_is_valid_shape(const std::array<size_t, 4> &shape) {
        // Check if the rows & cols are shaped like an ADRT output
        return (adrt::_impl::all_positive(shape) &&
                (std::get<1>(shape) == 4u) &&
                (std::get<3>(shape) <= adrt::_impl::max_size) &&
                (std::get<2>(shape) == (std::get<3>(shape) * 2_uz - 1_uz)) &&
                (adrt::_impl::is_pow2(std::get<3>(shape))));
    }

    // Implementation for adrt
    bool adrt_step_is_valid_iter(const std::array<size_t, 4> &shape, int iter) {
        return iter >= 0 && iter < adrt::num_iters(std::get<3>(shape));
    }

    // Implementation for adrt
    std::array<size_t, 5> adrt_buffer_shape(const std::array<size_t, 3> &shape) {
        return {
            std::get<0>(shape),
            4,
            std::get<1>(shape),
            1,
            2_uz * std::get<2>(shape) - 1_uz, // No overflow because n^2 fits in size_t, so must 2*n
        };
    }

    // Implementation for adrt
    std::array<size_t, 4> adrt_result_shape(const std::array<size_t, 3> &shape) {
        return {
            std::get<0>(shape),
            4,
            2_uz * std::get<2>(shape) - 1_uz, // No overflow because n^2 fits in size_t, so must 2*n
            std::get<1>(shape)
        };
    }

    std::array<size_t, 4> adrt_step_result_shape(const std::array<size_t, 4> &shape) {
        return shape;
    }

    // Implementation for bdrt
    bool bdrt_is_valid_shape(const std::array<size_t, 4> &shape) {
        return adrt::adrt_step_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_shape(const std::array<size_t, 4> &shape) {
        return adrt::bdrt_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_iter(const std::array<size_t, 4> &shape, int iter) {
        return adrt::adrt_step_is_valid_iter(shape, iter);
    }

    // Implementation for bdrt
    std::array<size_t, 5> bdrt_buffer_shape(const std::array<size_t, 4> &shape) {
        return {
            std::get<0>(shape), // batch
            4,  // quadrant
            std::get<3>(shape), // col
            1, // sections
            std::get<2>(shape), // row
        };
    }

    // Implementation for bdrt
    std::array<size_t, 4> bdrt_result_shape(const std::array<size_t, 4> &shape) {
        return shape;
    }

    std::array<size_t, 4> bdrt_step_result_shape(const std::array<size_t, 4> &shape) {
        return shape;
    }

    bool iadrt_is_valid_shape(const std::array<size_t, 4> &shape) {
        // bdrt also requires its input to have the shape of an adrt result, reuse
        return adrt::bdrt_is_valid_shape(shape);
    }

    std::array<size_t, 5> iadrt_buffer_shape(const std::array<size_t, 4> &shape) {
        return {
            std::get<0>(shape), // planes
            4, // Quadrants (shape[1])
            1,
            std::get<3>(shape), // N
            std::get<2>(shape), // 2 * N - 1
        };
    }

    std::array<size_t, 4> iadrt_result_shape(const std::array<size_t, 4> &shape) {
        return shape;
    }

    bool interp_adrtcart_is_valid_shape(const std::array<size_t, 4> &shape) {
        // bdrt also requires its input to have the shape of an adrt result, reuse
        return adrt::bdrt_is_valid_shape(shape);
    }

    std::array<size_t, 3> interp_adrtcart_result_shape(const std::array<size_t, 4> &shape) {
        return {
            std::get<0>(shape), // batch
            std::get<2>(shape), // rows
            4_uz * std::get<3>(shape), // cols. No overflow, merges quadrant and column dimensions
        };
    }

    bool fmg_restriction_is_valid_shape(const std::array<size_t, 4> &shape) {
        return (adrt::_impl::all_positive(shape) &&
                std::get<1>(shape) == 4u &&
                std::get<3>(shape) <= adrt::_impl::max_size &&
                std::get<2>(shape) == (std::get<3>(shape) * 2_uz - 1_uz) &&
                std::get<3>(shape) >= 2u &&
                std::get<3>(shape) % 2_uz == 0u);
    }

    std::array<size_t, 4> fmg_restriction_result_shape(const std::array<size_t, 4> &shape) {
        return {
            std::get<0>(shape), // batches
            4, // quadrants
            std::get<3>(shape) - 1_uz, // rows
            std::get<3>(shape) / 2_uz, // cols (halved)
        };
    }

    bool fmg_prolongation_is_valid_shape(const std::array<size_t, 3> &shape) {
        const size_t prl_max_size = adrt::_common::floor_div2(std::numeric_limits<size_t>::max());
        return (adrt::_impl::all_positive(shape) &&
                std::get<1>(shape) <= prl_max_size &&
                std::get<2>(shape) <= prl_max_size);
    }

    std::array<size_t, 3> fmg_prolongation_result_shape(const std::array<size_t, 3> &shape) {
        return {
            std::get<0>(shape), // batch
            2_uz * std::get<1>(shape), // rows. No overflow, checked in fmg_prolongation_is_valid_shape
            2_uz * std::get<2>(shape), // cols
        };
    }

    bool fmg_highpass_is_valid_shape(const std::array<size_t, 3> &shape) {
        return (adrt::_impl::all_positive(shape) &&
                std::get<1>(shape) >= 2u &&
                std::get<2>(shape) >= 2u);
    }

    std::array<size_t, 3> fmg_highpass_result_shape(const std::array<size_t, 3> &shape) {
        return shape;
    }

} // End namespace adrt
