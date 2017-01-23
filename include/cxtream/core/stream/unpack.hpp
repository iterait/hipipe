/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_UNPACK_HPP
#define CXTREAM_CORE_STREAM_UNPACK_HPP

#include <cxtream/core/stream/template_arguments.hpp>
#include <cxtream/core/utility/tuple.hpp>
#include <cxtream/core/utility/vector.hpp>

#include <range/v3/view/move.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/to_container.hpp>

namespace cxtream::stream {

namespace detail {

    // replace the columns in the stream with their values (i.e., raw batches)
    template<typename... FromColumns>
    constexpr auto unpack_columns()
    {
        return ranges::view::transform([](auto source) {
            auto proj = [](auto& column) { return std::move(column.value()); };
            auto subtuple = utility::tuple_type_view<FromColumns...>(source);
            return utility::tuple_transform(std::move(subtuple), std::move(proj));
        });
    }

    template<std::size_t Dim, bool OnlyOne, typename... FromColumns>
    struct unpack_impl
    {
        template<typename Rng>
        static constexpr auto impl(Rng range_of_tuples)
        {
            // project the selected columns to their values (i.e., batches)
            auto raw_range_of_tuples = range_of_tuples | unpack_columns<FromColumns...>();
            auto tuple_of_batches = utility::unzip(std::move(raw_range_of_tuples));
            // flatten the values in each column upto the given dimension
            return utility::tuple_transform(std::move(tuple_of_batches), [](auto&& batch_range) {
                // make sure to convert the flat view to std::vector to avoid dangling ref
                return utility::flat_view<Dim+1>(batch_range)
                  | ranges::view::move | ranges::to_vector;
            });
        }
    };

    // if only a single column is unpacked, do not return a tuple
    template<std::size_t Dim, typename FromColumn>
    struct unpack_impl<Dim, true, FromColumn>
    {
        template<typename Rng>
        static constexpr auto impl(Rng&& range_of_tuples)
        {
            return std::get<0>(
              unpack_impl<Dim, false, FromColumn>::impl(std::forward<Rng>(range_of_tuples)));
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Unpack a stream into a tuple of ranges.
///
/// This operation transforms the stream (i.e., a range of tuples of columns) into a
/// tuple of the types represented by the columns. The data can be unpacked in a specific
/// dimension and then the higher dimensions are joined together.
///
/// If there is only a single column to be unpacked, the result is an std::vector of the
/// corresponding type. If there are multiple columns to be unpacked, the result is a tuple of
/// std::vectors.
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(id, int)
///     CXTREAM_DEFINE_COLUMN(values, std::vector<double>)
///
///     std::vector<std::tuple<int, std::vector<double>>> data = {{3, {5., 7.}}, {1, {2., 4.}}};
///     auto rng = data | create<id, values>(4);
///
///     // unpack in the first dimesion
///     std::vector<int> unp_ids;
///     std::vector<std::vector<double>> unp_values;
///     std::tie(unp_ids, unp_values) = unpack(rng, from<id, values>);
///     // unp_ids == {3, 1}
///     // unp_values == {{5., 7.}, {2., 4.}}
///
///     // unpack a single column in the second dimesion
///     std::vector<double> unp_values_dim2;
///     unp_values_dim2 = unpack(rng, from<values>, dim<2>);
///     // unp_values_dim2 == {5., 7., 2., 4.}
/// \endcode
template<typename Rng, typename... FromColumns, int Dim = 1>
constexpr auto unpack(Rng&& rng, from_t<FromColumns...> f, dim_t<Dim> d = dim_t<1>{})
{
    return detail::unpack_impl<Dim, (sizeof...(FromColumns)==1), FromColumns...>::impl(
      std::forward<Rng>(rng));
}

}  // end namespace cxtream::stream
#endif
