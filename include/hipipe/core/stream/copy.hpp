/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/core/stream/transform.hpp>

namespace hipipe::stream {

/// \ingroup Stream
/// \brief Copy the data from FromColumns to the respective ToColumns.
///
/// The data from i-th FromColumn are copied to i-th ToColumn.
/// Note that the ToColumns examples must be constructible
/// from their FromColumns counterparts.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(i, int)
///     HIPIPE_DEFINE_COLUMN(i2, int)
///     HIPIPE_DEFINE_COLUMN(i3, int)
///     HIPIPE_DEFINE_COLUMN(l, long)
///
///     // rng is a stream with four identical columns
///     auto rng = view::iota(0, 10) | create<i>() |
///       copy(from<i>, to<i2>) | copy(from<i, i2>, to<i3, l>);
/// \endcode
///
/// \param from_cols The source columns.
/// \param to_cols The target columns.
template <typename... FromColumns, typename... ToColumns>
auto copy(from_t<FromColumns...> from_cols, to_t<ToColumns...> to_cols)
{
    static_assert(sizeof...(FromColumns) == sizeof...(ToColumns),
      "hipipe::stream::copy requires the same number of source and target columns.");

    static_assert(
      ((std::is_constructible_v<typename FromColumns::example_type,
                                const typename ToColumns::example_type&>) && ...),
      "hipipe::stream::copy target columns must be constructible "
      "from the respective source columns.");

    return stream::transform(from_cols, to_cols,
      [](const typename FromColumns::example_type&... vals) {
          return utility::maybe_tuple<typename ToColumns::example_type...>(vals...);
      });
}

}  // end namespace hipipe::stream
