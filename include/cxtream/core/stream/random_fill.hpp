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

#include <cxtream/core/stream/transform.hpp>
#include <cxtream/core/utility/random.hpp>
#include <cxtream/core/utility/vector.hpp>

#include <random>

namespace cxtream::stream {

namespace detail {

    // Create a random_fill transformation function that can be sent to stream::transform.
    template<typename FromColumn, typename ToColumn, typename Prng, typename Dist>
    struct wrap_random_fill_fun_for_transform {
        long rnddims;
        Dist dist;
        std::reference_wrapper<Prng> prng;

        typename ToColumn::batch_type operator()(typename FromColumn::batch_type& source)
        {
            using SourceVector = typename FromColumn::batch_type;
            using TargetVector = typename ToColumn::batch_type;
            constexpr long SourceDims = utility::ndims<SourceVector>::value;
            constexpr long TargetDims = utility::ndims<TargetVector>::value;
            static_assert(TargetDims <= SourceDims, "stream::random_fill requires"
              " the target column to have at most the same number of dimensions"
              " as the source column.");
            // get the size of the source up to the dimension of the target
            std::vector<std::vector<long>> target_size = utility::ndim_size<TargetDims>(source);
            // create, resize, and fill the target with random values
            TargetVector target;
            utility::ndim_resize(target, target_size);
            // make a copy of the stored distribution to avoid races when buffering
            utility::random_fill(target, rnddims, Dist{dist}, prng.get());
            return target;
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Fill the selected column of a stream with random values.
///
/// This function uses \ref utility::random_fill(). Furthermore, the column to be filled
/// is first resized so that it has the same size as the selected source column.
///
/// The selected `from` column has to be a multidimensional range with the number of
/// dimensions at least as large as the `to` column (i.e., the column to be filled).
///
/// Note: If there is no column the size could be taken from, than just resize
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
/// \param dist The random distribution to be used.
/// \param gen The random generator to be used.
template<typename FromColumn, typename ToColumn, typename Prng = std::mt19937,
         typename Dist = std::uniform_real_distribution<double>>
constexpr auto random_fill(from_t<FromColumn> size_from,
                           to_t<ToColumn> fill_to,
                           long rnddims = std::numeric_limits<long>::max(),
                           Dist dist = Dist{0, 1},
                           Prng& gen = cxtream::utility::random_generator)
{
    detail::wrap_random_fill_fun_for_transform<FromColumn, ToColumn, Prng, Dist>
      trans_fun{rnddims, std::move(dist), gen};
    return stream::transform(from<FromColumn>, to<ToColumn>, std::move(trans_fun), dim<0>);
}

}  // namespace cxtream::stream
#endif
