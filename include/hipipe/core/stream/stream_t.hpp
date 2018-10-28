/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/core/stream/batch_t.hpp>

#include <range/v3/view/any_view.hpp>

namespace hipipe::stream {


/// \ingroup Stream
/// The stream itself, i.e., a range of batches.
///
/// Unless specified otherwise, the stream transformers expect this type
/// and return this type. Exceptions are e.g. \ref stream::rebatch.
using forward_stream_t = ranges::any_view<batch_t, ranges::category::forward>;


/// \ingroup Stream
/// The stream type after special eager operations.
///
/// For instance, stream::rebatch reduces the stream to InputRange and
/// returns this type. Stream of such type cannot be further transformed.
using input_stream_t = ranges::any_view<batch_t, ranges::category::input>;

}  // namespace hipipe::stream
