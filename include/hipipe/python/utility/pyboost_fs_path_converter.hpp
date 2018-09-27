/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef HIPIPE_PYTHON_UTILITY_PYBOOST_FS_PATH_CONVERTER_HPP
#define HIPIPE_PYTHON_UTILITY_PYBOOST_FS_PATH_CONVERTER_HPP

#include <boost/python.hpp>

#include <experimental/filesystem>

namespace hipipe::python::utility {

struct fs_path_to_python_str {
    static PyObject* convert(const std::experimental::filesystem::path& path);
};

struct fs_path_from_python_str {
    fs_path_from_python_str();

    static void* convertible(PyObject* obj_ptr);

    static void construct(PyObject* obj_ptr,
                          boost::python::converter::rvalue_from_python_stage1_data* data);
};

}  // namespace hipipe::python::utility
#endif
