/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_UTILITY_PYBOOST_IS_REGISTERED_HPP
#define CXTREAM_PYTHON_UTILITY_PYBOOST_IS_REGISTERED_HPP

#include <boost/python.hpp>

namespace cxtream::python::utility {

/// \ingroup Python
/// \brief Check whether a converter for the given C++ class is registered in boost::python.
///
/// Beware that this function will report false for primitive types such as `int` or `double`.
template<typename T>
bool is_registered()
{
    namespace py = boost::python;
    py::type_info info = py::type_id<T>();
    const py::converter::registration* reg = py::converter::registry::query(info);
    return reg != nullptr && reg->m_to_python != nullptr;
}

}  // end namespace cxtream::python::utility
#endif
