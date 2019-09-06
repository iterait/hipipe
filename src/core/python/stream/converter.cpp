/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/build_config.hpp>
#ifdef HIPIPE_BUILD_PYTHON

#include <hipipe/core/python/range.hpp>
#include <hipipe/core/python/stream/converter.hpp>

#include <range/v3/view/transform.hpp>

namespace hipipe::python::stream {


range<ranges::any_view<boost::python::dict>> to_python(hipipe::stream::input_stream_t stream)
{
    ranges::any_view<boost::python::dict> range_of_dicts =
      ranges::views::transform(std::move(stream), &hipipe::stream::batch_t::to_python);

    // make python iterator out of the range of python types
    return range<ranges::any_view<boost::python::dict>>{std::move(range_of_dicts)};
}


}  // end namespace hipipe::python::stream

#endif  // HIPIPE_BUILD_PYTHON
