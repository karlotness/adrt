/*
 * Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
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

#ifdef _MSC_VER
// MSVC intrinsics
#include <intrin.h>
#endif

#include "adrt_cdefs_common.hpp"

namespace adrt { namespace _impl { namespace {

bool is_pow2(size_t val) {
    if(val == 0) {
        return false;
    }
    return (val & (val - 1)) == 0;
}

int num_iters_fallback(size_t shape) {
    // Relies on earlier check that shape != 0
    ADRT_ASSERT(shape != 0)
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    int r = 0;
    while(shape != 0) {
        ++r;
        shape >>= 1;
    }
    return r + (is_power_of_two ? 0 : 1) - 1;
}

#if !defined(__GNUC__) && !defined(__clang__)
// Fallback only needed if no GCC intrinsics
bool mul_check_fallback(size_t a, size_t b, size_t &prod) {
    prod = a * b;
    const bool overflow = (b != 0) && (a > std::numeric_limits<size_t>::max() / b);
    return !overflow;
}
#endif

template<size_t N>
bool all_positive(const std::array<size_t, N> &shape) {
    // Make sure all shapes are nonzero
    for(size_t i = 0; i < shape.size(); ++i) {
        if(shape[i] <= 0) {
            return false;
        }
    }
    return true;
}

// Implementation of adrt_num_iters

#if defined(__GNUC__) || defined(__clang__) // GCC intrinsics

int num_iters(size_t shape) {
    // Relies on earlier check that shape != 0
    ADRT_ASSERT(shape != 0)
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned int>::max()) {
        unsigned int ushape = static_cast<unsigned int>(shape);
        int lead_zero = __builtin_clz(ushape);
        return (std::numeric_limits<unsigned int>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long>::max()) {
        unsigned long ushape = static_cast<unsigned long>(shape);
        int lead_zero = __builtin_clzl(ushape);
        return (std::numeric_limits<unsigned long>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long long>::max()) {
        unsigned long long ushape = static_cast<unsigned long long>(shape);
        int lead_zero = __builtin_clzll(ushape);
        return (std::numeric_limits<unsigned long long>::digits - 1) - lead_zero + (is_power_of_two ? 0 : 1);
    }
    return adrt::_impl::num_iters_fallback(shape);
}

bool mul_check(size_t a, size_t b, size_t &prod) {
    const bool overflow = __builtin_mul_overflow(a, b, &prod);
    return !overflow;
}

#elif defined(_MSC_VER) // MSVC intrinsics

int num_iters(size_t shape) {
    // Relies on earlier check that shape != 0
    ADRT_ASSERT(shape != 0)
    const bool is_power_of_two = adrt::_impl::is_pow2(shape);
    if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned long>::max()) {
        unsigned long index;
        unsigned long ushape = static_cast<unsigned long>(shape);
        _BitScanReverse(&index, ushape);
        return index + (is_power_of_two ? 0 : 1);
    }

    #if defined(_M_X64) || defined(_M_ARM64)
    else if(std::numeric_limits<size_t>::max() <= std::numeric_limits<unsigned __int64>::max()) {
        unsigned long index;
        unsigned __int64 ushape = static_cast<unsigned __int64>(shape);
        _BitScanReverse64(&index, ushape);
        return index + (is_power_of_two ? 0 : 1);
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
        if(shape <= 1) {
            return 0;
        }
        return adrt::_impl::num_iters(shape);
    }

    namespace _common {
        adrt::_common::Optional<size_t> mul_check(size_t a, size_t b) {
            adrt::_common::Optional<size_t> prod;
            const bool ok = adrt::_impl::mul_check(a, b, *prod);
            prod.set_ok(ok);
            return prod;
        }

    } // End adrt::_common

    // Implementation for adrt
    bool adrt_is_valid_shape(const std::array<size_t, 3> &shape) {
        // Make sure array is square
        return (adrt::_impl::all_positive(shape) && // All entries must be nonzero
                (std::get<1>(shape) == std::get<2>(shape)) && // Must be square
                (adrt::_impl::is_pow2(std::get<2>(shape)))); // Must have power of two shape
    }

    // Implementation for adrt
    bool adrt_step_is_valid_shape(const std::array<size_t, 4> &shape) {
        // Check if the rows & cols are shaped like an ADRT output
        return (adrt::_impl::all_positive(shape) &&
                (std::get<1>(shape) == 4) &&
                (std::get<2>(shape) == (std::get<3>(shape) * 2 - 1)) &&
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
            2 * std::get<2>(shape) - 1, // No overflow because n^2 fits in size_t, so must 2*n
        };
    }

    // Implementation for adrt
    std::array<size_t, 4> adrt_result_shape(const std::array<size_t, 3> &shape) {
        return {
            std::get<0>(shape),
            4,
            2 * std::get<2>(shape) - 1, // No overflow because n^2 fits in size_t, so must 2*n
            std::get<1>(shape)
        };
    }

    // Implementation for bdrt
    bool bdrt_is_valid_shape(const std::array<size_t, 4> &shape) {
        return adrt_step_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_shape(const std::array<size_t, 4> &shape) {
        return bdrt_is_valid_shape(shape);
    }

    bool bdrt_step_is_valid_iter(const std::array<size_t, 4> &shape, int iter) {
        return adrt_step_is_valid_iter(shape, iter);
    }

    // Implementation for bdrt
    std::array<size_t, 5> bdrt_buffer_shape(const std::array<size_t, 4> &shape) {
        return {
            std::get<0>(shape), // batch
            4,  // quadrant
            std::get<2>(shape), // row
            std::get<3>(shape), // col
            1 // sections
        };
    }

    // Implementation for bdrt
    std::array<size_t, 4> bdrt_result_shape(const std::array<size_t, 4> &shape) {
        return shape;
    }

    bool iadrt_is_valid_shape(const std::array<size_t, 4> &shape) {
        // bdrt also requires its input to have the shape of an adrt result, reuse
        return adrt::bdrt_is_valid_shape(shape);
    }

    std::array<size_t, 5> iadrt_buffer_shape(const std::array<size_t, 4> &shape) {
        return {
            4, // Quadrants (shape[1])
            std::get<0>(shape), // planes
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
            4 * std::get<3>(shape), // cols. No overflow, merges quadrant and column dimensions
        };
    }

} // End namespace adrt
