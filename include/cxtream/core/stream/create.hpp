/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_CREATE_HPP
#define CXTREAM_CORE_STREAM_CREATE_HPP

#include <cxtream/core/utility/tuple.hpp>

#include <range/v3/view/transform.hpp>
#include <range/v3/view/chunk.hpp>

namespace cxtream::stream {

/// \ingroup Stream
/// \brief Converts a range to a stream (i.e., to a range of tuples of columns).
///
/// The value type of the input range is supposed
/// to be either the type represented by the column to be created,
/// or a tuple of such types if there are more columns to be created.
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(id, int)
///     CXTREAM_DEFINE_COLUMN(age, int)
///
///     // rng is a stream where each batch is a single element from 0..9
///     auto rng = view::iota(0, 10) | create<id>();
///
///     // batched_rng is a stream with a single batch with numbers 0..9
///     auto rng = view::iota(0, 10) | create<id>(50);
///
///     // also multiple columns can be created at once
///     auto rng = view::zip(view::iota(0, 10), view::iota(30, 50)) | create<id, age>();
/// \endcode
///
/// \param batch_size The requested batch size for the provided data.
template<typename... Columns>
constexpr auto create(std::size_t batch_size = 1)
{
    assert(batch_size >= 1);
    return ranges::view::chunk(batch_size)
      | ranges::view::transform([](auto&& source) {
            return std::tuple<Columns...>{
              utility::unzip_if<(sizeof...(Columns) > 1)>(std::forward<decltype(source)>(source))};
        });
}

} // end namespace cxtream::stream
#endif
