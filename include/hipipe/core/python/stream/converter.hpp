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

#include <hipipe/core/stream/stream_t.hpp>

#include <range/v3/view/any_view.hpp>

namespace hipipe::python::stream {


/// \ingroup Python
/// \brief Make a Python \ref range from a stream (i.e, a view of batches).
///
/// This turns a view of batches to a Python's generator of dicts.
range<ranges::any_view<boost::python::dict>> to_python(hipipe::stream::input_stream_t stream);


}  // end namespace hipipe::python::stream

#endif  // HIPIPE_BUILD_PYTHON
