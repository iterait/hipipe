/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_PAD_HPP
#define CXTREAM_CORE_STREAM_PAD_HPP

#include <cxtream/core/stream/transform.hpp>
#include <cxtream/core/utility/vector.hpp>

namespace cxtream::stream {
namespace detail {

    // Create a transformation function that can be sent to stream::transform.
    template<typename FromColumn, typename MaskColumn, typename ValT>
    struct wrap_pad_fun_for_transform {
        ValT value;

        std::tuple<typename FromColumn::batch_type, typename MaskColumn::batch_type>
        operator()(typename FromColumn::batch_type& source)
        {
            using SourceVector = typename FromColumn::batch_type;
            using MaskVector = typename MaskColumn::batch_type;
            constexpr long SourceDims = utility::ndims<SourceVector>::value;
            constexpr long MaskDims = utility::ndims<MaskVector>::value;
            static_assert(MaskDims <= SourceDims, "stream::pad requires"
              " the number of padded dimensions (i.e., the number of dimensions"
              " of the mask) to be at most the number of dimensions of the source column.");
            // create the positive mask
            std::vector<std::vector<long>> source_size = utility::ndim_size<MaskDims>(source);
            MaskVector mask;
            utility::ndim_resize<MaskDims>(mask, source_size, true);
            // pad the source
            utility::ndim_pad<MaskDims>(source, value);
            // create the negative mask
            source_size = utility::ndim_size<MaskDims>(source);
            utility::ndim_resize<MaskDims>(mask, source_size, false);
            return {std::move(source), std::move(mask)};
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Pad the selected column to a rectangular size.
///
/// Each batch is padded separately.
///
/// The mask of the padded values is created along with the
/// padding. The mask evaluates to `true` on the positions with the original
/// elements and to `false` on the positions of the padded elements.
/// The mask column should be a multidimensional vector of type
/// bool/char/int/... The dimensionality of the mask column is used to deduce
/// how many dimensions should be padded in the source column.
///
/// This transformer internally uses \ref utility::ndim_pad().
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(sequences, std::vector<int>)
///     CXTREAM_DEFINE_COLUMN(sequence_masks, std::vector<bool>)
///     std::vector<std::vector<int>> data = {{1, 2}, {3, 4, 5}, {}, {6, 7}};
///     auto rng = data
///       | create<sequences>(2)
///       | pad(from<sequences>, mask<sequence_masks>, -1);
///     // sequences_batch_1 == {{1, 2, -1}, {3, 4, 5}} 
///     // sequences_batch_2 == {{-1, -1}, {6, 7}} 
///     // sequence_masks_batch_1 == {{true, true, false}, {true, true, true}} 
///     // sequence_masks_batch_2 == {{false, false}, {true, true}} 
/// \endcode
///
/// \param f The column to be padded.
/// \param m The column where the mask should be stored and from which the dimension
///          is taken.
/// \param value The value to pad with.
template<
  typename FromColumn, typename MaskColumn,
  // The value type is automatically deduced as the type of the source column
  // in the dimension of the mask column.
  typename ValT =
    typename utility::ndim_type_t<
      typename FromColumn::batch_type, 
      utility::ndims<typename MaskColumn::batch_type>::value>>
constexpr auto pad(from_t<FromColumn> f, mask_t<MaskColumn> m, ValT value = ValT{})
{
    detail::wrap_pad_fun_for_transform<FromColumn, MaskColumn, ValT>
      trans_fun{std::move(value)};
    return stream::transform(from<FromColumn>, to<FromColumn, MaskColumn>,
                             std::move(trans_fun), dim<0>);
}

}  // namespace cxtream::stream
#endif
