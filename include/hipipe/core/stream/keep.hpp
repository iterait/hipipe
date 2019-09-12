/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/core/stream/stream_t.hpp>

#include <range/v3/view/transform.hpp>


namespace hipipe::stream {

namespace rgv = ranges::views;

namespace detail {

    template<typename... Columns>
    class keep_fn {
    private:
        friend rgv::view_access;

        static auto bind(keep_fn<Columns...> fun)
        {
            return ranges::make_pipeable(std::bind(fun, std::placeholders::_1));
        }

    public:
        CPP_template(typename Rng)(requires ranges::input_range<Rng>)
        forward_stream_t operator()(Rng&& rng) const
        {
            return rgv::transform(std::forward<Rng>(rng),
              [](batch_t batch) -> batch_t {
                  batch_t result;
                  (result.raw_insert_or_assign<Columns>(std::move(batch.at<Columns>())), ...);
                  return result;
            });
        }

        /// \cond
        CPP_template(typename Rng)(requires !ranges::input_range<Rng>)
        void operator()(Rng&&) const
        {
            CONCEPT_ASSERT_MSG(ranges::input_range<Rng>(),
              "stream::keep only works on ranges satisfying the input_range concept.");
        }
        /// \endcond
    };

}  // namespace detail


/// \ingroup Stream
/// \brief Keep the specified columns in the stream, drop everything else.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(value, double)
///     std::vector<std::tuple<int, double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data | create<id, value>() | keep<value>;  // now it has only the value column
/// \endcode
template <typename... Columns>
rgv::view<detail::keep_fn<Columns...>> keep{};

}  // end namespace hipipe::stream
