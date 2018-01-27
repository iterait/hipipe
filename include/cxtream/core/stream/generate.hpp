/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_GENERATE_HPP
#define CXTREAM_CORE_STREAM_GENERATE_HPP

#include <cxtream/core/stream/transform.hpp>
#include <cxtream/core/utility/vector.hpp>

namespace cxtream::stream {
namespace detail {

    // Create a transformation function that can be sent to stream::transform.
    template<typename FromColumn, typename ToColumn, typename Gen, int Dim>
    struct wrap_generate_fun_for_transform {
        Gen gen;
        long gendims;

        typename ToColumn::batch_type operator()(typename FromColumn::batch_type& source)
        {
            using SourceVector = typename FromColumn::batch_type;
            using TargetVector = typename ToColumn::batch_type;
            constexpr long SourceDims = utility::ndims<SourceVector>::value;
            static_assert(Dim <= SourceDims, "stream::generate requires"
              " the dimension in which to apply the generator to be at most the number"
              " of dimensions of the source column (i.e., the column the shape is taken"
              " from).");
            // get the size of the source up to the dimension of the target
            std::vector<std::vector<long>> target_size = utility::ndim_size<Dim>(source);
            // create, resize, and fill the target using the generator
            TargetVector target;
            utility::ndim_resize<Dim>(target, target_size);
            utility::generate<Dim>(target, gen, gendims);
            return target;
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Fill the selected column using a generator (i.e., a nullary function).
///
/// This function uses \ref utility::generate(). Furthermore, the column to be filled
/// is first resized so that it has the same size as the selected source column.
///
/// Tip: If there is no column the size could be taken from, than just resize
/// the target column manually and use it as both `from` column and `to` column.
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(id, int)
///     CXTREAM_DEFINE_COLUMN(value, double)
///     std::vector<int> data = {3, 1, 2};
///     auto rng = data
///       | create<id>()
///       // assign each id a value from an increasing sequence
///       | generate(from<id>, to<value>, [i = 0]() mutable { return i++; });
/// \endcode
///
/// \param size_from The column whose size will be used to initialize the generated column.
/// \param fill_to The column to be filled using the generator.
/// \param gen The generator to be used.
/// \param gendims The number of generated dimensions. See \ref utility::generate().
/// \param d This is the dimension in which will the generator be applied.
///          E.g., if set to 1, the generator result is considered to be a single example.
///          The default is ndims<ToColumn::batch_type> - ndims<gen()>. This value
///          has to be positive.
template<typename FromColumn, typename ToColumn, typename Gen,
         int Dim = utility::ndims<typename ToColumn::batch_type>::value
                 - utility::ndims<std::result_of_t<Gen()>>::value>
constexpr auto generate(from_t<FromColumn> size_from,
                        to_t<ToColumn> fill_to,
                        Gen gen,
                        long gendims = std::numeric_limits<long>::max(),
                        dim_t<Dim> d = dim_t<Dim>{})
{
    detail::wrap_generate_fun_for_transform<FromColumn, ToColumn, Gen, Dim>
      trans_fun{std::move(gen), gendims};
    return stream::transform(from<FromColumn>, to<ToColumn>, std::move(trans_fun), dim<0>);
}

}  // namespace cxtream::stream
#endif
