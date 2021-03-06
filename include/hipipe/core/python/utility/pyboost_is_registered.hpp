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

#include <boost/python.hpp>

namespace hipipe::python::utility {

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

}  // end namespace hipipe::python::utility

#endif  // HIPIPE_BUILD_PYTHON
