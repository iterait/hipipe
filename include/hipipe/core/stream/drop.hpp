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

#include <range/v3/view/transform.hpp>


namespace hipipe::stream {

namespace detail {

    template<typename... Columns>
    class drop_fn {
    private:
        friend ranges::view::view_access;

        static auto bind(drop_fn<Columns...> fun)
        {
            return ranges::make_pipeable(std::bind(fun, std::placeholders::_1));
        }

    public:
        CPP_template(class Rng)(requires ranges::InputRange<Rng>)
        forward_stream_t operator()(Rng&& rng) const
        {
            return ranges::view::transform(std::forward<Rng>(rng),
              [](batch_t batch) -> batch_t {
                  ((batch.erase<Columns>()), ...);
                  return batch;
            });
        }

        /// \cond
        CPP_template(class Rng)(requires !ranges::InputRange<Rng>)
        void operator()(Rng&&) const
        {
            CONCEPT_ASSERT_MSG(ranges::InputRange<Rng>(),
              "stream::drop only works on ranges satisfying the InputRange concept.");
        }
        /// \endcond
    };

}  // namespace detail


/// \ingroup Stream
/// \brief Drops columns from a stream.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(value, double)
///     std::vector<std::tuple<int, double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data | create<id, value>() | drop<id>;
/// \endcode
template <typename... Columns>
ranges::view::view<detail::drop_fn<Columns...>> drop{};

}  // end namespace hipipe::stream
