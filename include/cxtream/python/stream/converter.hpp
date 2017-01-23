/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_STREAM_CONVERTER_HPP
#define CXTREAM_PYTHON_STREAM_CONVERTER_HPP

#include <cxtream/python/range.hpp>
#include <cxtream/python/utility/pyboost_column_converter.hpp>

#include <range/v3/view/transform.hpp>

namespace cxtream::python::stream {

/// \ingroup Python
/// \brief Make a Python \ref range from a stream (i.e, a range of tuples of cxtream columns).
///
/// Only a view of the given range is created, therefore, the given range
/// cannot be an rvalue of a container.
///
/// Tuples of columns are converted using columns_to_python().
template<typename Rng>
auto to_python(Rng&& rng)
{
    // by forwarding rng to view::transform, we make sure that it is
    // not an lvalue of a container
    auto range_of_dicts = std::forward<Rng>(rng)
      // transform the range of columns to a range of python types
      | ranges::view::transform([](auto&& tuple) {
            return utility::columns_to_python(std::forward<decltype(tuple)>(tuple));
        });

    // make python iterator out of the range of python types
    using PyRng = decltype(range_of_dicts);
    return range<PyRng>{std::move(range_of_dicts)};
}

}  // end namespace cxtream::python::stream
#endif
