/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_UTILITY_PYBOOST_NDARRAY_CONVERTER_HPP
#define CXTREAM_PYTHON_UTILITY_PYBOOST_NDARRAY_CONVERTER_HPP

// this header has to be included before the numpy header
#include <cxtream/python/initialize.hpp>

#include <Python.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>

#include <stdexcept>
#include <vector>

namespace cxtream::python::utility {

namespace detail {

    template<typename T>
    struct to_ndarray_trait
    {
        using type_t = PyObject*;

        static PyObject* convert(T val)
        {
            boost::python::object obj{std::move(val)};
            Py_INCREF(obj.ptr());
            return obj.ptr();
        }

        static int typenum()
        {
            return NPY_OBJECT;
        }
    };

#define CXTREAM_DEFINE_TO_NDARRAY_TRAIT(C_TYPE, NP_TYPE, TYPENUM)      \
    template<>                                                         \
    struct to_ndarray_trait<C_TYPE>                                    \
    {                                                                  \
        using type_t = NP_TYPE;                                        \
                                                                       \
        static NP_TYPE convert(C_TYPE val)                             \
        {                                                              \
            return val;                                                \
        }                                                              \
                                                                       \
        static int typenum()                                           \
        {                                                              \
            return TYPENUM;                                            \
        }                                                              \
    }

    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(bool,          npy_bool,       NPY_BOOL);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::int8_t,   npy_byte,       NPY_BYTE);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::uint8_t,  npy_ubyte,      NPY_UBYTE);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::int16_t,  npy_int16,      NPY_INT16);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::uint16_t, npy_uint16,     NPY_UINT16);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::int32_t,  npy_int32,      NPY_INT32);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::uint32_t, npy_uint32,     NPY_UINT32);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::int64_t,  npy_int64,      NPY_INT64);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(std::uint64_t, npy_uint64,     NPY_UINT64);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(float,         npy_float,      NPY_FLOAT);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(double,        npy_double,     NPY_DOUBLE);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(long double,   npy_longdouble, NPY_LONGDOUBLE);
    CXTREAM_DEFINE_TO_NDARRAY_TRAIT(PyObject*,     PyObject*,      NPY_OBJECT);

#undef CXTREAM_DEFINE_TO_NDARRAY_TRAIT

    /// \ingroup Python
    /// \brief Convert the given variable to the corresponding ndarray type.
    ///
    /// If the variable is not one of the selected builtin types, it is
    /// converted using boost::python::object.
    template<typename T>
    auto to_ndarray_element(T val)
    {
        return detail::to_ndarray_trait<T>::convert(std::move(val));
    }

    /// \ingroup Python
    /// \brief Get the ndarray type corresponding to the given C++ type.
    template<typename T>
    using ndarray_type_t = typename detail::to_ndarray_trait<T>::type_t;

    /// \ingroup Python
    /// \brief Get the ndarray type number corresponding to the given C++ type.
    template<typename T>
    int to_ndarray_typenum()
    {
        return detail::to_ndarray_trait<T>::typenum();
    }

}  // namespace detail

/// \ingroup Python
/// \brief Build ndarray from a one dimensional std::vector.
template<typename T>
PyObject* to_ndarray(const std::vector<T>& vec)
{
    auto data = std::make_unique<detail::ndarray_type_t<T>[]>(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) data[i] = detail::to_ndarray_element(vec[i]);
    npy_intp dims[1]{static_cast<npy_intp>(vec.size())};

    PyObject* arr = PyArray_SimpleNewFromData(
      1, dims, detail::to_ndarray_typenum<T>(), reinterpret_cast<void*>(data.release()));
    if (!arr) throw std::runtime_error{"Cannot create Python NumPy ndarray."};
    // we have to tell NumPy to delete the data when the array is removed
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(arr), NPY_ARRAY_OWNDATA);
    return arr;
}

}  // namespace cxtream::python::utility
#endif
