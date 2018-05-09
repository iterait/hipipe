/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_RANDOM_FILL_HPP
#define CXTREAM_CORE_STREAM_RANDOM_FILL_HPP

#include <cxtream/core/stream/generate.hpp>

#include <functional>
#include <random>

namespace cxtream::stream {

/// \ingroup Stream
/// \brief Fill the selected column of a stream with random values.
///
/// This function uses \ref stream::generate() and has a similar semantics.
/// That is, the column to be filled is first resized so that it has the same
/// size as the selected source column.
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
///       | random_fill(from<id>, to<value>);
///       | transform(from<id, value>, [](...){ ... });
/// \endcode
///
/// \param size_from The column whose size will be used to initialize the random column.
/// \param fill_to The column to be filled with random data.
/// \param rnddims The number of random dimensions. See \ref utility::random_fill().
/// \param dist The random distribution to be used. This object is copied on every
///             use to avoid race conditions with \ref stream::buffer().
/// \param prng The random generator to be used.
/// \param d This is the dimension in which will the generator be applied.
///          E.g., if set to 1, the generator result is considered to be a single example.
///          The default is ndims<ToColumn::batch_type> - ndims<dist(prng)>.
///          This value has to be positive.
template<typename FromColumn, typename ToColumn, typename Prng = std::mt19937,
         typename Dist = std::uniform_real_distribution<double>,
         int Dim = utility::ndims<typename ToColumn::batch_type>::value
                 - utility::ndims<std::result_of_t<Dist(Prng&)>>::value>
constexpr auto random_fill(from_t<FromColumn> size_from,
                           to_t<ToColumn> fill_to,
                           long rnddims = std::numeric_limits<long>::max(),
                           Dist dist = Dist{0, 1},
                           Prng& prng = cxtream::utility::random_generator,
                           dim_t<Dim> d = dim_t<Dim>{})
{
    // distribution is always copied to avoid race conditions
    auto fun = [dist, &prng]() { return std::invoke(Dist{dist}, prng); };
    return stream::generate(size_from, fill_to, std::move(fun), rnddims, d);
}

}  // namespace cxtream::stream
#endif
