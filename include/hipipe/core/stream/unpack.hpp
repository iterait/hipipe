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

#include <hipipe/core/stream/template_arguments.hpp>
#include <hipipe/core/utility/tuple.hpp>
#include <hipipe/core/utility/vector.hpp>

#include <range/v3/view/move.hpp>
#include <range/v3/view/transform.hpp>

namespace hipipe::stream {

namespace detail {


    template<std::size_t Dim, bool OnlyOne, typename... FromColumns>
    struct unpack_impl
    {
        template<typename Rng>
        static auto unpack_columns(Rng&& rng)
        {
            return ranges::view::transform(std::forward<Rng>(rng), [](batch_t source)
                -> std::tuple<typename FromColumns::batch_type...> {
                    return {std::move(source.extract<FromColumns>())...};
            });
        }

        template<typename Rng>
        static
        std::tuple<std::vector<utility::ndim_type_t<typename FromColumns::batch_type, Dim>>...>
        impl(Rng&& range_of_batches)
        {
            static_assert(std::is_same_v<std::decay_t<ranges::range_value_type_t<Rng>>, batch_t>,
              "hipipe::stream::unpack requires a range of batches as input.");
            static_assert(((Dim <= utility::ndims<typename FromColumns::batch_type>::value) && ...),
              "hipipe::stream::unpack requires the requested dimension to be less or equal to the"
              " dimension of all the unpacked columns.");
            auto raw_range_of_tuples = unpack_columns(std::forward<Rng>(range_of_batches));
            auto tuple_of_batches = utility::unzip(std::move(raw_range_of_tuples));
            // flatten the values in each column upto the given dimension
            return utility::tuple_transform(std::move(tuple_of_batches), [](auto&& batch_range) {
                // make sure to convert the flat view to std::vector to avoid dangling ref
                return ranges::view::move(utility::flat_view<Dim+1>(batch_range));
            });
        }
    };

    // if only a single column is unpacked, do not return a tuple
    template<std::size_t Dim, typename FromColumn>
    struct unpack_impl<Dim, true, FromColumn>
    {
        template<typename Rng>
        static
        std::vector<utility::ndim_type_t<typename FromColumn::batch_type, Dim>>
        impl(Rng&& range_of_tuples)
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
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(values, std::vector<double>)
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
auto
// The return type is actually the following, but GCC won't handle it.
// utility::maybe_tuple<std::vector<utility::ndim_type_t<typename FromColumns::batch_type, Dim>>...>
unpack(Rng&& rng, from_t<FromColumns...> f, dim_t<Dim> d = dim_t<1>{})
{
    return detail::unpack_impl<Dim, (sizeof...(FromColumns)==1), FromColumns...>::impl(
      std::forward<Rng>(rng));
}

}  // end namespace hipipe::stream
