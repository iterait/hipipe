/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/core/stream/stream_t.hpp>
#include <hipipe/core/utility/tuple.hpp>

#include <range/v3/functional/bind_back.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/chunk.hpp>

#include <typeinfo>
#include <unordered_map>

namespace hipipe::stream {

namespace rg = ranges;
namespace rgv = ranges::views;

namespace detail {

    template<typename... Columns>
    struct create_impl {

        template<typename Source>
        batch_t operator()(Source&& source) const
        {
            batch_t batch;

            if constexpr(sizeof...(Columns) == 0) {
                static_assert("hipipe::stream::create: At least one column has to be provided.");
            } else if constexpr(sizeof...(Columns) == 1) {
                static_assert(std::is_constructible_v<Columns..., Source&&>,
                  "hipipe::stream::create: "
                  "Cannot convert the given data range to the selected column type.");
                batch.insert_or_assign<Columns...>(rg::to_vector(std::forward<Source>(source)));
            } else {
                using SourceValue = rg::range_value_t<Source>;
                static_assert(std::is_constructible_v<
                  std::tuple<typename Columns::example_type...>, SourceValue&&>,
                  "hipipe::stream::create: "
                  "Cannot convert the given data range to the selected column types.");
                std::tuple<Columns...> data = utility::unzip(std::forward<Source>(source));
                utility::tuple_for_each(data, [&batch](auto& column){
                    batch.insert_or_assign<std::decay_t<decltype(column)>>(std::move(column));
                });
            }

            return batch;
        }
    };

    template<typename... Columns>
    class create_fn {
    public:
        CPP_template(class Rng)(requires rg::forward_range<Rng>)
        forward_stream_t operator()(Rng&& rng, std::size_t batch_size = 1) const
        {
            return rgv::transform(
              rgv::chunk(std::forward<Rng>(rng), batch_size),
              create_impl<Columns...>{});
        }

        auto operator()(std::size_t batch_size = 1) const
        {
            return rg::make_view_closure(rg::bind_back(create_fn{}, batch_size));
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Converts a data range to a HiPipe stream.
///
/// The value type of the input range is supposed
/// to be either the type represented by the column to be created,
/// or a tuple of such types if there are more columns to be created.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(age, int)
///
///     // rng is a stream where each batch is a single element from 0..9
///     auto rng = views::iota(0, 10) | create<id>();
///
///     // batched_rng is a stream with a single batch with numbers 0..9
///     auto rng = views::iota(0, 10) | create<id>(50);
///
///     // also multiple columns can be created at once
///     auto rng = views::zip(views::iota(0, 10), views::iota(30, 50)) | create<id, age>();
/// \endcode
///
/// \param batch_size The requested batch size of the new stream.
template<typename... Columns>
rgv::view_closure<detail::create_fn<Columns...>> create{};

} // end namespace hipipe::stream
