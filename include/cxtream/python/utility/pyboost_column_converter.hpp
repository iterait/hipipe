/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_UTILITY_PYBOOST_COLUMN_CONVERTER_HPP
#define CXTREAM_PYTHON_UTILITY_PYBOOST_COLUMN_CONVERTER_HPP

#include <cxtream/core/utility/tuple.hpp>
#include <cxtream/python/range.hpp>
#include <cxtream/python/utility/pyboost_ndarray_converter.hpp>

#include <boost/python.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace cxtream::python::utility {

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

/// \ingroup Python
/// \brief Convert a tuple of cxtream columns into a Python `dict`.
///
/// The dict is indexed by `column.name` and the value is `column.value`.
/// The values (i.e, the batches) are converted to Python lists using to_python().
/// If the batch is a multidimensional std::vector<std::vector<...>>, it
/// is converted to multidimensional Python list.
template<typename Tuple>
boost::python::dict columns_to_python(Tuple tuple)
{
    boost::python::dict res;
    cxtream::utility::tuple_for_each(tuple, [&res](auto& column) {
        res[column.name()] = to_python(std::move(column.value()));
    });
    return res;
}

}  // namespace cxtream::python::utility
#endif
