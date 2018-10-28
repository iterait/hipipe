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
#include <hipipe/build_config.hpp>
#ifdef HIPIPE_BUILD_PYTHON

#include <hipipe/core/stream/column.hpp>
#include <hipipe/core/python/range.hpp>

#include <range/v3/view/transform.hpp>

namespace hipipe::python::stream {

/// \ingroup Python
/// \brief Make a Python \ref range from a stream (i.e, a view of batches).
/// TODO compiled function
inline auto to_python(hipipe::stream::input_stream_t stream)
{
    auto range_of_dicts =
      ranges::view::transform(std::move(stream), &hipipe::stream::batch_t::to_python);

    // make python iterator out of the range of python types
    using PyRng = decltype(range_of_dicts);
    return range<PyRng>{std::move(range_of_dicts)};
}

}  // end namespace hipipe::python::stream

#endif  // HIPIPE_BUILD_PYTHON
