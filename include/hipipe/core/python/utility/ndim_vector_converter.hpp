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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <filesystem>

#include <boost/python.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace hipipe::python::utility {

// recursive transformation from a multidimensional vector to python //

namespace detail {
    template <typename T>
    pybind11::object impl(T val);

    template <typename T>
    pybind11::object impl(std::unique_ptr<T> ptr);

    template <typename T>
    pybind11::object impl(T val)
    {
        pybind11::object obj = pybind11::cast(std::move(val));
        return obj;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
    pybind11::object impl(std::vector<T> vec)
    {
        pybind11::object obj = pybind11::array_t<T>(vec.size(), std::move(vec.data()));
        return obj;
    }

    template <typename T, std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
    pybind11::object impl(std::vector<T> vec)
    {
        pybind11::list l;
        for (std::size_t i=0; i<vec.size(); ++i) {
            l.append(detail::impl(std::move(vec[i])));
        }
        return l;
    }

    template <typename T>
    pybind11::object impl(std::unique_ptr<T> ptr)
    {
        return detail::impl(std::move(*ptr));
    }

    template <>
    inline pybind11::object impl(std::filesystem::path p)
    {
        return pybind11::cast(p.string());
    }

    template <>
    inline pybind11::object impl(std::vector<bool> vec)
    {
        pybind11::array_t<bool> arr(vec.size());
        bool* arr_data = (bool*)(arr.request().ptr);
        for (std::size_t i=0; i<vec.size(); ++i) {
            arr_data[i] = vec[i];
        }
        return arr;
    }

    template <>
    inline pybind11::object impl(cv::Point2f pt) {
        return pybind11::make_tuple(pt.x, pt.y);
    }

    template <>
    inline pybind11::object impl(cv::Mat m) {   
        /// copy of https://github.com/pybind/pybind11/issues/538

        std::string format = pybind11::format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int dim;
        switch(m.type()) {
            case CV_8U:
                format = pybind11::format_descriptor<unsigned char>::format();
                elemsize = sizeof(unsigned char);
                dim = 2;
                break;
            case CV_8UC3:
                format = pybind11::format_descriptor<unsigned char>::format();
                elemsize = sizeof(unsigned char);
                dim = 3;
                break;
            case CV_32F:
                format = pybind11::format_descriptor<float>::format();
                elemsize = sizeof(float);
                dim = 2;
                break;
            case CV_64F:
                format = pybind11::format_descriptor<double>::format();
                elemsize = sizeof(double);
                dim = 2;
                break;
            default:
                throw std::logic_error("Unsupported type");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) m.rows, (size_t) m.cols};
            strides = {elemsize * (size_t) m.cols, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
            strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
        }
        return pybind11::array(pybind11::buffer_info(
            m.data,         /* Pointer to buffer */
            elemsize,       /* Size of one scalar */
            format,         /* Python struct-style format descriptor */
            dim,            /* Number of dimensions */
            bufferdim,      /* Buffer dimensions */
            strides         /* Strides (in bytes) for each index */
            ));
    }
}  // namespace detail


/// \ingroup Python	
/// \brief Create a Python list-like object out of a multidimensional std::vector.	
///	
/// If the vector is multidimensional, i.e., std::vector<std::vector<...>>,	
/// the resulting Python structure will be multidimensional as well.
template<typename T>
pybind11::object to_python(std::vector<T> v)
{
    pybind11::object obj = detail::impl(std::move(v));
    return obj;
}


}  // namespace hipipe::python::utility

#endif  // HIPIPE_BUILD_PYTHON
