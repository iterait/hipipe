/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_TEMPLATE_ARGUMENTS_HPP
#define CXTREAM_CORE_STREAM_TEMPLATE_ARGUMENTS_HPP

#include <functional>

namespace cxtream::stream {

template <typename... Columns>
struct from_t {
};

/// Helper type representing columns which should be transformed.
template <typename... Columns>
auto from = from_t<Columns...>{};

template <typename... Columns>
struct to_t {
};

/// Helper type representing columns to which should a transformation save the result.
template <typename... Columns>
auto to = to_t<Columns...>{};

template <typename... Columns>
struct by_t {
};

/// Helper type representing columns by which one should filter etc.
template <typename... Columns>
auto by = by_t<Columns...>{};

template <typename... Columns>
struct cond_t {
};

/// Helper type representing boolean column denoting whether transformation should be applied.
template <typename... Columns>
auto cond = cond_t<Columns...>{};

template <int Dim>
struct dim_t {
};

/// Helper type representing dimension.
template <int Dim>
auto dim = dim_t<Dim>{};

struct identity_t {
    template <typename T>
    constexpr T&& operator()(T&& val) const noexcept
    {
        return std::forward<T>(val);
    }
};

/// Function object type forwarding the given object back to the caller.
auto identity = identity_t{};

struct ref_wrap_t {
    template <typename T>
    constexpr decltype(auto) operator()(T& val) const noexcept
    {
        return std::ref(val);
    }
};

/// Function object type wrapping the given object in std::reference_wrapper.
auto ref_wrap = ref_wrap_t{};

}  // namespace cxtream::stream
#endif
