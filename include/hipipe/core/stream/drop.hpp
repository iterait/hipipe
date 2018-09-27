/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef HIPIPE_CORE_STREAM_DROP_HPP
#define HIPIPE_CORE_STREAM_DROP_HPP

#include <hipipe/core/utility/tuple.hpp>

#include <range/v3/view/transform.hpp>

namespace hipipe::stream {

namespace detail {

    template<typename Source, typename... DropColumns>
    struct drop_impl;

    template<typename... SourceColumns, typename... DropColumns>
    struct drop_impl<std::tuple<SourceColumns...>, DropColumns...> {

        constexpr decltype(auto) operator()(std::tuple<SourceColumns...> source) const
        {
            return utility::tuple_remove<DropColumns...>(std::move(source));
        }

    };

    template<typename... Columns>
    class drop_fn {
    private:
        friend ranges::view::view_access;

        static auto bind(drop_fn<Columns...> fun)
        {
            return ranges::make_pipeable(std::bind(fun, std::placeholders::_1));
        }

    public:
        template<typename Rng, CONCEPT_REQUIRES_(ranges::ForwardRange<Rng>())>
        constexpr auto operator()(Rng&& rng) const
        {
            using StreamType = ranges::range_value_type_t<Rng>;
            return ranges::view::transform(
              std::forward<Rng>(rng),
              drop_impl<StreamType, Columns...>{});
        }

        /// \cond
        template<typename Rng, CONCEPT_REQUIRES_(!ranges::ForwardRange<Rng>())>
        void operator()(Rng&&) const
        {
            CONCEPT_ASSERT_MSG(ranges::ForwardRange<Rng>(),
              "stream::drop only works on ranges satisfying the ForwardRange concept.");
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
constexpr ranges::view::view<detail::drop_fn<Columns...>> drop{};

}  // end namespace hipipe::stream
#endif
