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

#include <hipipe/core/python/range.hpp>
#include <hipipe/core/python/utility/vector_converter.hpp>

#include <boost/python.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace hipipe::python::utility {

// recursive transformation from a multidimensional vector to python //

namespace detail {

    // conversion of std::vector to a Python list-like type //
    // Vectors of builtin primitive types (e.g., int, bool, ...) are converted
    // to numpy ndarrays. Vector of other types are converted to lists and
    // their elements are converted using boost::python::object

    template<typename T>
    struct vector_to_python_impl {
        static PyObject* impl(T val)
        {
            boost::python::object obj{std::move(val)};
            Py_INCREF(obj.ptr());
            return obj.ptr();
        }
    };

    template<typename T>
    struct vector_to_python_impl<std::vector<T>> {
        static PyObject* impl(std::vector<T> vec)
        {
            if (std::is_arithmetic<T>{}) {
                return utility::to_ndarray(std::move(vec));
            }

            PyObject* list{PyList_New(vec.size())};
            if (!list) throw std::runtime_error{"Unable to create Python list."};
            for (std::size_t i = 0; i < vec.size(); ++i) {
                PyList_SET_ITEM(list, i, vector_to_python_impl<T>::impl(std::move(vec[i])));
            }
            return list;
        }
    };

}  // namespace detail

/// \ingroup Python
/// \brief Create a Python list-like object out of a multidimensional std::vector.
///
/// If the vector is multidimensional, i.e., std::vector<std::vector<...>>,
/// the resulting Python structure will be multidimensional as well.
template<typename T>
boost::python::object to_python(std::vector<T> v)
{
    namespace py = boost::python;
    py::handle<> py_obj_handle{detail::vector_to_python_impl<std::vector<T>>::impl(std::move(v))};
    return py::object{py_obj_handle};
}

}  // namespace hipipe::python::utility

#endif  // HIPIPE_BUILD_PYTHON
