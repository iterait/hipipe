/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_DROP_HPP
#define CXTREAM_CORE_STREAM_DROP_HPP

#include <cxtream/core/utility/tuple.hpp>

#include <range/v3/view/transform.hpp>

namespace cxtream::stream {

template<typename... Columns>
constexpr auto drop_fn()
{
    return ranges::view::transform([](auto&& source) {
        return utility::tuple_remove<Columns...>(std::forward<decltype(source)>(source));
    });
}

/// \ingroup Stream
/// \brief Drops columns from a stream.
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(id, int)
///     CXTREAM_DEFINE_COLUMN(value, double)
///     std::vector<std::tuple<int, double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data | create<id, value>() | drop<id>;
/// \endcode
template <typename... Columns>
auto drop = drop_fn<Columns...>();

}  // end namespace cxtream::stream
#endif
