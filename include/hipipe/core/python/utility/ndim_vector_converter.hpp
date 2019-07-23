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
#include <iostream>
#include <typeinfo>
#include <opencv2/core/core.hpp>
#include <filesystem>

#include <boost/python.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace hipipe::python::utility {

// recursive transformation from a multidimensional vector to python //

namespace detail {
    // Workaround for T=fs::path, because template specialization causes internal compiler error
    template<typename T>
    pybind11::object convert_path(T val) {
        std::cout << "Converting fs::path" << std::endl;
        return pybind11::cast(val.string());
    }

    template <typename T>
    pybind11::object impl(T val);

    template <typename T>
    pybind11::object impl(std::unique_ptr<T> ptr);

    template <typename T>
    pybind11::object impl(T val)
    {
        if constexpr (std::is_same<T, std::filesystem::path>::value) {
            return convert_path(val);
        }
        std::cout << "Converting base of type: " <<  typeid(val).name() << std::endl;
        pybind11::object obj = pybind11::cast(val);
        return obj;
    }

    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
    pybind11::object impl(std::vector<T> vec) {
        std::cout << "Converting to numpy array" << std::endl;
        pybind11::object obj = pybind11::array_t<T>(vec.size(), vec.data());
        return obj;
    }

    template <typename T, std::enable_if_t<!std::is_arithmetic<T>::value, int> = 0>
    pybind11::object impl(std::vector<T> vec)
    {
        std::cout << "Converting list" << std::endl;
        pybind11::list l;
        for (std::size_t i=0; i<vec.size(); ++i) {
            std::cout << "Converting list element" << std::endl;
            l.append(detail::impl(std::move(vec[i])));
        }
        return l;
    }

    template <typename T>
    pybind11::object impl(std::unique_ptr<T> ptr)
    {
        std::cout << "Converting unique_ptr" << std::endl;
        return detail::impl(*ptr);
    }

//TODO: this causes internal compiler error
/*
    template <>
    inline pybind11::object impl(std::filesystem::path p) {
        std::cout << "Converting filesystem::path" << std::endl;
        return pybind11::cast(p.string());
    }
*/

    template <>
    inline pybind11::object impl(std::vector<bool> vec) {
        pybind11::array_t<bool> arr(vec.size());
        bool* arr_data = (bool*)(arr.request().ptr);
        for (std::size_t i=0; i<vec.size(); ++i) {
            arr_data[i] = vec[i];
        }
        return arr;
    }

    template <>
    inline pybind11::object impl(cv::Point2f pt) {
        std::cout << "Converting cv::Point" << std::endl;
        return pybind11::make_tuple(pt.x, pt.y);
    }

    template <>
    inline pybind11::object impl(cv::Mat m) {   
        std::cout << "Converting cv::Mat" << std::endl;

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
}


template<typename T>
pybind11::object to_python(std::vector<T> v)
{
    std::cout << "Converting vector of type: " << typeid(v).name() << std::endl;
    pybind11::object obj = detail::impl(std::move(v));
    std::cout << "Conversion done" << std::endl;
    return obj;
}


}  // namespace hipipe::python::utility

#endif  // HIPIPE_BUILD_PYTHON
